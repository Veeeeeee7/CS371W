# Reference library — OOD × GIS project

40 papers, grouped by the role they play in the project. Filenames are
`Short-Name__arXiv-ID.pdf`. ★ = read first.

Companion file: **`LITERATURE-REVIEW.md`** — one paragraph per paper covering
motivation, method, results and limitations, plus a synthesis of what the 40
together establish. This file is the *why it matters here* index; that one is the
papers on their own terms.

Note: `refs/ood.pdf` is your original copy of the OODSelect paper; the same paper
is also in `01-core-ood-method/` under a descriptive name.

---

## 01-core-ood-method — the method being ported, and the line it sits on

| ★ | Paper | arXiv | Why it matters here |
|---|---|---|---|
| ★ | **Aggregation Hides OOD Generalization Failures from Spurious Correlations** (Salaudeen, Zhang, Alhamoud, Beery, Ghassemi, MIT, NeurIPS 2025) | 2510.24884 | The anchor. Introduces accuracy-on-the-inverse-line (AoTIL) and OODSelect. §on VLM/LLM descriptions failing is the gap this project fills. |
| ★ | **Accuracy on the Line** (Miller et al. 2021) | 2107.04649 | The phenomenon being disputed: ID accuracy strongly predicts OOD accuracy. Establishes the probit transform used throughout. |
| ★ | **ID and OOD Performance Are Sometimes Inversely Correlated on Real-world Datasets** (Teney et al. 2022) | 2209.00613 | The direct precursor to AoTIL — shows inverse correlation appears once the model population is diverse enough. Read for how to build a model population. |
| | **Accuracy on the Wrong Line: pitfalls of noisy data** (2024) | 2406.19049 | Alternative explanation for when the line breaks — label noise. A confound you need to rule out. |
| | **Measuring Robustness to Natural Distribution Shifts** (Taori et al. 2020) | 2007.00644 | Origin of the large-model-population methodology. |
| | **A Survey on Evaluation of OOD Generalization** (2024) | 2403.01874 | Orientation / related-work scaffolding. |
| | **Distributionally Robust Neural Networks for Group Shifts (Group DRO)** (Sagawa et al. 2019) | 1911.08731 | The standard mitigation once you have groups. Relevant if you want a "so what do we do about it" section. |

## 02-geospatial-ood — what the GIS field already knows about distribution shift

| ★ | Paper | arXiv | Why it matters here |
|---|---|---|---|
| ★ | **OT on the Map: Quantifying Domain Shifts in Geographic Space (GeoSpOT)** (Zhang, Betti, Klemmer, Rolf, Alvarez-Melis) | 2604.16220 | Optimal-transport distance between geospatial domains that predicts cross-region transfer difficulty. Closest existing work — and Haoran Zhang is a co-author on the OODSelect paper too. Your natural point of contact / possible collaborator. |
| ★ | **EarthShift: benchmark for robustness to real-world distribution shifts in EO** (ASU, 2026) | 2605.29330 | The current benchmark for geospatial OOD. Note that it measures *aggregate* degradation — exactly what OODSelect argues is misleading. |
| | **Beyond Accuracy: Calibration of GeoFMs and Sensitivity to Distribution Shifts** (2026) | 2608.16614 | Argues averaged-rank evaluation of geo models is too narrow. Adjacent argument to yours. |
| | **No One Knows the State of the Art in Geospatial Foundation Models** (2026) | 2605.12678 | Evaluation-practice critique in the geo field. Good for motivation. |
| | **GeoBS: Information-Theoretic Quantification of Geographic Bias** (2025) | 2509.23482 | How geographic bias is currently measured. Your spatial-structure metrics should be positioned against this. |
| | **Geographic Bias and Diversity in AI Evaluation** (2026) | 2606.05187 | Framing / motivation. |
| | **Quantifying geographic domain shift: transferability of mobility flow models** (2026) | 2608.21567 | Cross-region transferability on vector/network GIS data. |
| | **Benchmarking GeoFMs for Agriculture** (2026) | 2606.29664 | Geographic transferability tested systematically. |

## 03-slice-discovery — the "explain the sets" half

| ★ | Paper | arXiv | Why it matters here |
|---|---|---|---|
| ★ | **Differential Subgroup Discovery: Characterizing Where Two Populations Differ, and Why** (2026) | 2604.27741 | Directly the stage-2 problem: given two populations (selected vs. not), describe the difference interpretably. |
| ★ | **Discovery and Spatial Characterisation of Multiple Shortcut Groups** (KCL, 2026) | 2608.14051 | The nearest thing to your idea in the literature — but "spatial" there means *within-image* pixel space, not geographic space. Read carefully: it's both your closest relative and your clearest point of differentiation. |
| ★ | **Domino: Discovering Systematic Errors with Cross-Modal Embeddings** (Stanford 2022) | 2203.14960 | The canonical slice-discovery method and the standard baseline. |
| | **The Spotlight** (d'Eon et al. 2021) | 2107.00758 | Early continuous-relaxation approach to finding failure regions — structurally similar to OODSelect's relaxation. |
| | **Slice Finder / Automated Data Slicing for Model Validation** (2018) | 1807.06068 | Feature-based slicing on *tabular* data. Most directly transferable baseline for your tabular setting. |
| | **GEORGE: No Subclass Left Behind** (2020) | 2011.12945 | Hidden stratification; recovering unlabeled subgroups. |
| | **CB-SLICE: Concept-Based Interpretable Error Slice Discovery** (2026) | 2605.29836 | Current SOTA for making discovered slices human-readable. |
| | **LADDER: Language-Driven Slice Discovery** (2024) | 2408.07832 | The LLM-description approach the OODSelect authors found unreliable. |
| | **Error Discovery by Clustering Influence Embeddings** (2023) | 2312.04712 | Formalizes *coherence* of a discovered slice — useful metric definition. |
| | **Detecting Systematic Weaknesses along Predefined Human-Understandable Dimensions** (2025) | 2502.12360 | Slice discovery when you already have named attributes — which, with GIS layers, you do. |
| | **Discovering Latent Groups for Robust Classification** (2026) | 2606.23609 | Group discovery without annotations. |

## 04-datasets-benchmarks — candidate testbeds

| ★ | Paper | arXiv | Why it matters here |
|---|---|---|---|
| ★ | **Retiring Adult: New Datasets for Fair ML (Folktables)** (Ding et al. 2021) | 2108.04884 | **The primary dataset.** ACS person-level data, 50 states + PR, five prediction tasks, every feature named, and every row carries a PUMA code — so selections aggregate into something mappable. |
| ★ | **TableShift: Benchmarking Distribution Shift in Tabular Data** (2024) | 2312.07577 | Non-geographic tabular shifts — your control condition for "is this geography-specific or just tabular?" |
| ★ | **WILDS** (Koh et al. 2021) | 2012.07421 | Benchmark suite OODSelect ran on; PovertyMap-WILDS is the imagery fallback if you ever extend. |
| | **In Search of Lost Domain Generalization (DomainBed)** (2020) | 2007.01434 | The other suite OODSelect ran on. |
| | **GEO-Bench-2** (IBM et al. 2025) | 2511.15658 | Current geospatial evaluation standard. |
| | **GEO-Bench: Toward Foundation Models for Earth Monitoring** (2023) | 2306.03831 | Predecessor; the tasks are the field's reference set. |
| | **SustainBench** (2021) | 2111.04724 | Geo-tagged socioeconomic prediction tasks. |
| | **OOD Performance of Tabular Foundation Models** (2026) | 2607.26000 | Recent tabular OOD evidence; informs the model-population design. |

## 05-geoai-explainability — the spatial statistics you need to do stage 2 honestly

| ★ | Paper | arXiv | Why it matters here |
|---|---|---|---|
| ★ | **Beyond spatial and random cross-validation: prediction-domain adaptive evaluation** (2026) | 2605.13689 | Why random CV lies on spatial data. Determines how you validate the stage-2 explainer. |
| ★ | **Spatial Confounding: A review** (2026) | 2602.17792 | The central threat to any "these covariates explain the subset" claim. Read before writing stage 2. |
| ★ | **Do Location Encoders Capture Spatial Effects? A GeoShapley Benchmark** (2026) | 2606.23453 | GeoShapley decomposes a prediction into location vs. feature contributions — a ready-made tool for separating "where" from "why". |
| | **Target-Weighted Cross-Validation for spatial prediction** (2026) | 2603.29981 | Concrete CV scheme aligned with the deployment domain. |
| | **Reframing Spatial Dependence as Geographic Feature Attribution** (TUM/MIT 2025) | 2506.16996 | Bridges spatial autocorrelation and feature attribution. |
| | **GeoAggregator explainability** (2025) | 2507.17977 | Transformer for geospatial *tabular* data — the architecture family closest to your setting. |

---

## Suggested reading order (first pass, ~6 papers)

1. `2510.24884` — OODSelect. Focus on Algorithm 1, the baselines, and the limitations section.
2. `2209.00613` — Teney et al., for how to build a diverse model population.
3. `2108.04884` — Folktables, for the data and the state-level splits.
4. `2604.16220` — GeoSpOT, for how the geo field currently quantifies domain shift.
5. `2608.14051` — the shortcut-groups paper, to sharpen what's different about your "spatial".
6. `2605.13689` — spatial CV, so stage 2 is defensible from the start.
