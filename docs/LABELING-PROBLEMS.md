# Known problems with labeling in flood susceptibility mapping

Related-work foundation. All DOIs verified against Crossref. Numbers marked ✅
were read from the primary PDF; others come from publisher article pages —
re-check before a figure becomes load-bearing in a write-up.

**The headline:** label choices move AUC by 0.13–0.41. Conditioning-factor and
algorithm choices move it by 0.02–0.04. Labels dominate by roughly an order of
magnitude, and **nobody has run that comparison in the flood domain.**

---

## 1. The one sentence that describes this project's central risk

> **"Most erroneous predictions, but highest predictive performances, were
> obtained from models generated with highly incomplete inventories and
> predictors that were able to directly describe the respective
> incompleteness."**
>
> — Steger, S., Brenning, A., Bell, R., & Glade, T. (2017). The influence of
> systematically incomplete shallow landslide inventories on statistical
> susceptibility models and suggestions for improvements. *Landslides* 14(5),
> 1767–1781. DOI `10.1007/s10346-017-0820-0`

Written about landslides in Austria, but it is our failure mode exactly. If the
label is FEMA SFHA, and FEMA mapping effort correlates with population,
development and county wealth, then **any predictor that proxies for "where
FEMA mapped" — land cover, imperviousness, road density, building density —
will let the model learn the mapping process instead of flood physics, and score
beautifully while doing it.**

Their measured effect: simulating 80% deletion of landslides in forested areas
dropped the slope odds ratio per 10° from **7.3 [5.5–9.8] to 5.0 [3.7–7.0]** on
real data, and **5.4 [4.8–6.1] to 4.0 [3.5–4.5]** on synthetic. The physical
relationship degrades while performance does not. Mixed-effects models were most
robust in their tests.

---

## 2. Positive-label sampling — directly relevant to our sampler

**Tehrany, M. Sh. & Jones, S. (2017).** Evaluating the variations in the flood
susceptibility maps accuracies due to the alterations in the type and extent of
the flood inventory. *ISPRS Archives* XLII-4/W5, 209–214.
DOI `10.5194/isprs-archives-XLII-4-W5-209-2017`

Same flood event (2011 Brisbane), same logistic regression, same predictors.
Only the representation of the positive label changed:

| positive representation | prediction-rate AUC |
|---|---|
| polygon | 63% |
| 1000 random points | 76% |
| **700 random points** | **88%** |
| 500 random points | 80% |
| 300 random points | 74% |
| 100 random points | 71% |
| 50 random points | 65% |

**A 25-point AUC swing from how positives were sampled out of the same
footprint — and it is non-monotonic**, so it is not a sample-size curve, it is
an artifact.

Consequence for us: the points-per-polygon cap is not a tuning knob. Pick it on
a principled basis (minimum separation distance, area-proportional draw),
document it, and report sensitivity — never select it by maximizing AUC.

---

## 3. Negative sampling — the largest measured effect, and a trap

"Did not flood" is never observed. The sampling rule is a free parameter nobody
standardizes, and it moves results more than the model does.

**Gu, T., Duan, P., Wang, M., Li, J., & Zhang, Y. (2024).** Effects of
non-landslide sampling strategies on machine learning models in landslide
susceptibility mapping. *Scientific Reports* 14, 7201.
DOI `10.1038/s41598-024-57964-5`

| strategy | RF | SVM | CatBoost |
|---|---|---|---|
| PU bagging | 0.865 | 0.882 | 0.897 |
| buffer control (500 m) | 0.769 | 0.814 | 0.798 |
| **K-means clustering** | **0.923** | **0.915** | **0.924** |

K-means won every AUC comparison and produced a **degenerate map** — over 90% of
the area in high/very-high susceptibility, ~40% of susceptibility zones gone
from the output. Sampling choice alone moved AUC by **0.126–0.154** on identical
positives and predictors.

**Zhang, Y., Wei, Y., Yao, R., Sun, P., Zhen, N., & Xia, X. (2025).** Data
Uncertainty of Flood Susceptibility Using Non-Flood Samples. *Remote Sensing*
17(3), 375. DOI `10.3390/rs17030375`

Four non-flood datasets, same positives: ensemble ROC **0.71, 0.72, 0.94, 0.95**
— a **0.24 AUC spread from negative sampling alone.**

**Huang, H., Tao, Z., Zhan, J., & Wang, C. (2025).** Contrast or Diversity:
Non-Flood sampling in urban flood susceptibility modelling. *Journal of
Hydrology* 656, 133053. DOI `10.1016/j.jhydrol.2025.133053`

Names the mechanism: *"High sample contrast led to excellent binary
classification performance but resulted in an overestimation of flood
susceptibility"*; high diversity gave the reverse. **The classification metric
and the map quality move in opposite directions along the sampling parameter.**
You cannot tune negative sampling by maximizing AUC.

**Jia, Z., et al. (2026).** Non-landslide sample for landslide susceptibility
prediction modeling: a review of selection strategies. *JRMGE* 18(4),
2859–2880. DOI `10.1016/j.jrmge.2025.12.011` — the framing review; recommends
1:2 positive:negative and a 300 m buffer. Note its ranking favours the
strategies that make negatives *easiest to separate*, which is the trap above.

---

## 4. Inflated performance metrics

**Fleuchaus, P., et al. (2021).** Retrospective evaluation of landslide
susceptibility maps and review of validation practice. *Environmental Earth
Sciences* 80(15), 485. DOI `10.1007/s12665-021-09770-9`
— 50 studies: **73% reported AUC ≥ 80%; validation scores below 70% were never
observed.** A literature with a performance floor and no failures is not
measuring performance.

**Reichenbach, P., et al. (2018).** A review of statistically-based landslide
susceptibility models. *Earth-Science Reviews* 180, 60–91.
DOI `10.1016/j.earscirev.2018.03.001` ✅
— 565 articles, 1983–2016. **38.9% performed no prediction evaluation at all;
32.0% did not measure model fit; only 3.0% estimated uncertainty.**

**Kumar, C., Walton, G., Santi, P., & Luza, C. (2025).** Random Cross-Validation
Produces Biased Assessment of Machine Learning Performance in Regional Landslide
Susceptibility Prediction. *Remote Sensing* 17(2), 213. DOI `10.3390/rs17020213`
— **random CV runs 6–18% higher than spatial CV.** With C5.0, more training data
raised random-CV AUC 0.86 → 0.92 while spatial-CV rose only 0.82 → 0.86: more
data widened the illusion. **Complex models (RF, SVM) showed larger gaps than
simple ones (LR, LDA).**

**Ploton, P., et al. (2020).** Spatial validation reveals poor predictive
performance of large-scale ecological mapping models. *Nature Communications*
11, 4540. DOI `10.1038/s41467-020-18321-y`
— the canonical statement. Random k-fold R² = 0.53; spatial k-fold **R² = 0.14**.
Predictive ability *"virtually null beyond 100 km."*

**Karasiak, N., et al. (2022).** Spatial dependence between training and test
sets. *Machine Learning* 111(7), 2715–2740. DOI `10.1007/s10994-021-05972-1`
— classification analogue: **8.68 points of OA inflation**, worst at pixel level
where samples are near-duplicates. Directly applicable to sampling many points
from one polygon.

**Wadoux, A.M.J.-C., et al. (2021).** Spatial cross-validation is not the right
way to evaluate map accuracy. *Ecological Modelling* 457, 109692.
DOI `10.1016/j.ecolmodel.2021.109692` — cite for balance; the debate is live.

---

## 5. Label *definition* — the largest effect anyone has measured

**Lima, P., Steger, S., Glade, T., & Mergili, M. (2023).** Conventional
data-driven landslide susceptibility models may only tell us half of the story.
*Geomorphology* 430, 108638. DOI `10.1016/j.geomorph.2023.108638`

Three models differing *only* in what counts as a positive, scored against four
validation targets:

| validation target | Model R | Model B | Model R+R |
|---|---|---|---|
| full body | 0.71 | **0.83** | 0.77 |
| release areas | **0.87** | 0.76 | 0.57 |
| depositional areas | **0.46** | 0.81 | **0.84** |
| max runout point | 0.52 | **0.76** | 0.73 |

**Model R scores 0.87 or 0.46 — better than most published models, or worse than
a coin flip — depending only on which label definition you validate against.**

---

## 6. FEMA SFHA as ground truth

**Wing, O.E.J., et al. (2018).** Estimates of present and future flood risk in
the conterminous United States. *Environmental Research Letters* 13(3), 034023.
DOI `10.1088/1748-9326/aaac65` ✅
— *"Nearly 41 million Americans live within the 1% annual exceedance probability
floodplain (compared to only 13 million when calculated using FEMA flood
maps)."* Exposure **2.6–3.1× higher** than FEMA implies. Read as a label
statement: **the positive class is missing roughly two-thirds of true
positives**, concentrated on small streams and pluvial settings.

**Flores, A.B., Collins, T.W., Grineski, S.E., Amodeo, M., Porter, J., Sampson,
C.C., & Wing, O. (2025).** Federally-overlooked flood risk inequities in the
conterminous United States. *Scientific Reports* 15, 10678.
DOI `10.1038/s41598-025-95120-9`

**The most important paper here.** ~26 million residents and ~6 million
properties sit in modelled 100-year floodplains FEMA does not recognize —
18 million pluvial, 5.2 million fluvial, 3.1 million coastal. And the misses are
socially structured: a one-SD decrease in median household income associates
with a **5.2% rise** in overlooked at-risk residents (6%–16.7% by flood type);
Black residents face +4.6% overlooked pluvial risk in metro areas;
Hispanic/Latinx composition associates with **14.2%–20.8%** increases in
overlooked fluvial risk in suburban/rural contexts.

**The consequence nobody has drawn:** a model trained on NFHL inherits label
noise correlated with income and race. It is **not i.i.d., so it will not wash
out with scale**, and the resulting map is systematically wrong in exactly the
communities already underserved.

**Wu, A.N., Zhang, Y., & Stouffs, R. (2026).** Deep learning completes US flood
hazard maps revealing millions exposed to previously unrecognized risk. *Nature
Communications* 17, 5983. DOI `10.1038/s41467-026-74336-x`

**Our closest published precedent — read in full before finalizing the design.**
cGAN on 2023 NFHL + 3DEP, 300k tiles, 30 m output.

- FEMA has modelled only **one-third of US river channels**; only **a quarter of
  those** were updated within five years; **40% of 917 metro/micro areas remain
  inadequately mapped**
- They acknowledge training data mixed "high-quality, updated maps and older,
  outdated" records; mitigation was architectural
- In-sample **mIoU ≥ 0.73**; out-of-sample in morphologically different regions
  **mIoU < 0.50** — a ~0.23 transfer gap on FEMA labels
- **A single national model beat local models by +0.06 to +0.17 mIoU
  out-of-sample.** Geographic diversity in training bought generalization.

Note the circularity to confront: trained on NFHL, then used to argue NFHL is
wrong. That holds only if the terrain prior dominates the label noise — assumed,
not demonstrated.

---

## 7. Conditioning factors — the small term

**Reichenbach et al. (2018)** ✅ — **596 distinct input variables** across 565
papers; **445 (74.6%) appear only once or twice.** Models used 2–22 variables,
mean 9.

**Fardin, S.G., Zêzere, J.L., Mendes, T.S.G., & Simões, S.J.C. (2026).** When
Land Use/Land Cover Misleads. *GeoHazards* 7(3), 94.
DOI `10.3390/geohazards7030094`
— Adding LULC raised flood AUC from 0.71 to **0.85–0.89**, but the authors argue
the gain is confounded: informal settlements concentrate on steep slopes, so
LULC *"may partially reproduce the effect of slope already captured by
topographic variables."* **They recommend excluding LULC as a conditioning
factor**, repositioning it as exposure.

This is Steger 2017's mechanism appearing in the flood domain. **Given FEMA
labels, treat land cover, imperviousness, road density and building density as
suspect — they proxy for where FEMA does its mapping.**

**Sinčić, M., et al. (2025).** *NHESS* 25, 183–206. DOI `10.5194/nhess-25-183-2025`
— factor *classification* criteria dominated outcomes over statistical method
choice; AUC range ~84–88.5.

---

## 8. Transferability — thin, and the clearest open question

**Wu et al. (2026)** — in-sample mIoU ≥0.73 → out-of-sample <0.50, "localized
overfitting"; national pooling beat local fitting.

**Ploton et al. (2020)** — predictive ability "virtually null beyond 100 km."

**Khalid, R. & Khan, U.T. (2024).** *Geocarto International* 39(1), 2316653.
DOI `10.1080/10106049.2024.2316653` — a counter-example: ANN-SMOTE transferred
across four Ontario watersheds with accuracy retained. Weak transfer test
(one province), and absolute OA was low (0.549).

**No paper isolates label inconsistency as a cause of transfer failure.** Highly
plausible — FEMA vintage, contractor and methodology vary by county and state,
so a multi-state model faces label *definition* drift across boundaries, not
just covariate shift. Plausible is not cited. **This is directly testable with
our data: hold out by state, check whether transfer error correlates with NFHL
vintage and coverage differences between train and test states.**

---

## 9. Magnitude ordering — the argument to lead with

| what you vary | AUC effect | source |
|---|---|---|
| conditioning-factor classification | ~0.02–0.04 | Sinčić 2025 |
| algorithm | a few points | multiple |
| **negative sampling rule** | **0.13–0.24** | Gu 2024; Zhang 2025 |
| **positive-label representation** | **0.25** | Tehrany & Jones 2017 |
| **label definition** | **0.41** | Lima 2023 |

Label choices dominate by an order of magnitude. **That comparison has never
been run in the flood domain.**

---

## 10. Gaps this project is positioned to fill

1. **No flood-domain equivalent of Steger on inventory bias.** No study
   quantifies flood inventory spatial bias against reporting or population
   density.
2. **Nobody has connected Flores et al.'s structured FEMA error to ML label
   quality.** The argument — non-i.i.d. label noise that will not average out —
   is available, well-supported, and unmade.
3. **Nobody has isolated label inconsistency as a cause of geographic transfer
   failure.** We will have per-state NFHL vintage and coverage metadata, which
   is exactly the variable to regress transfer error against.

**One design note.** The literature's implied best experiment is: hold model and
predictors fixed; vary (a) label source, (b) negative sampling rule, (c)
validation scheme; report the AUC spread attributable to each. Our Zone A vs AE
design *is* a label-source variation — a within-NFHL one, which is cleaner than
comparing across products because everything else is held constant. Worth
framing it that way explicitly, and worth adding (b) and (c) as cheap
sensitivity axes since the sampler has to make those choices anyway.
