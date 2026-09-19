# What already exists — and what you still have to build

Companion to `DATA-SOURCES.md`. Verified 2026-09-16 by fetching and, where
possible, downloading and inspecting the actual data.

---

## The short answer

**No pre-assembled, per-location, multi-state US flood-susceptibility training
dataset exists.** Confirmed absent from Hugging Face, Kaggle, Zenodo, Figshare
and GitHub. Every deposit that joins flood labels to conditioning factors at
pixel grain is single-country-non-US (Pakistan, Malaysia, India, Ecuador) or
single-metro (Houston). You are building it.

**But roughly half the covariates are already computed and downloadable**, and
there is a published CONUS-scale paper whose assembly you can cite and copy.

---

## 1. The paper to cite and copy

> Woznicki, S.A., Baynes, J., Panlasigui, S., Mehaffey, M., Neale, A., 2019.
> Development of a spatially complete floodplain map of the conterminous United
> States using random forest. *Science of the Total Environment* 647:942–953.
> `10.1016/j.scitotenv.2018.07.353`

This is the closest published precedent to your assembly, at exactly your scale:

- **Response variable: FEMA SFHA.** Same label you're using.
- **Predictors:** 10 NLCD derivatives, a 30 m DEM, SSURGO/STATSGO2 — including
  **distance to nearest channel, slope, and soil taxonomy.**
- **Scale:** random forest fit across **202 HUC-4 units**, CONUS.
- **Reported performance:** overall hit rate 0.79 against FIRM, **worse in the
  arid Southwest.**
- Output distributed as EPA EnviroAtlas "Estimated Floodplain Map for the
  Conterminous United States," a 30 m binary raster.

Cite it as the methodological precedent for predicting FEMA zones from terrain
at national scale, and for the conditioning-factor selection.

### Two things about this paper that matter to your project specifically

**It partly anticipates your geographic analysis — but not your angle.** They
report regional performance variation (arid Southwest worst) across 202 HUC-4
units. That is prior art for "flood susceptibility models transfer unevenly
across US regions." What they did **not** do is stratify by Zone A vs Zone AE.
Your provenance question is untouched.

**Their result is a testable prediction of your hypothesis.** Their model
predicts FEMA's map from terrain. If Zone A boundaries are themselves largely a
terrain heuristic while Zone AE boundaries come from hydraulic modelling, then
their model should fit **Zone A areas better than Zone AE areas** — because in
Zone A it is imitating a function of the same inputs it has. Nobody has checked
this. It is a cheap, sharp, early experiment, and it can be run against a
published model rather than one you trained.

### Do not use their raster as your label

The EnviroAtlas floodplain map is itself an ML output from terrain covariates
that overlap yours. Training on it teaches your model to imitate their model.
Same caution applies to the Global Flood Susceptibility Map v1
(`zenodo.org/records/20568218`, 30 m global XGBoost output) — it is somebody
else's finished prediction, useful as a benchmark, circular as a label.

---

## 2. EPA StreamCat — the covariate backbone, and the real time saver

<https://www.epa.gov/national-aquatic-resource-surveys/streamcat-dataset>

> Hill, R.A., Weber, M.H., Leibowitz, S.G., Olsen, A.R., Thornbrugh, D.J., 2016.
> The Stream-Catchment (StreamCat) Dataset. *JAWRA* 52:120–128.
> `10.1111/1752-1688.12372`

**1,296 pre-computed metrics for all 2.65 million NHDPlus V2 catchments in
CONUS**, keyed by COMID, each available as local catchment (`...Cat`) or
accumulated upstream watershed (`...Ws`).

### What it gives you free

| Your covariate | StreamCat metric |
|---|---|
| Elevation | `elev[Cat\|Ws]` |
| **TWI** | `wetindex[Cat\|Ws]` — Composite Topographic Index. The real thing. |
| Rainfall | `precip8110`, `precip9120` (PRISM normals), annual variants |
| Land cover | 16 NLCD classes × 8 years |
| Imperviousness | `pctimp[year]` × 8 years, plus impervious-on-slope variants |
| Soil | `clay`, `sand`, `om`, `perm`, `rckdep`, `wtdep`, `kffact` (STATSGO) |
| Bonus | `runoff`, `bfi` (baseflow index), 18 lithology classes, `bankfullwidth`/`depth` |

### What it does not have

**No slope** (grep returns zero — only impervious-on-slope composites). No
curvature, no SPI, no HAND, no distance-to-river, no drainage density, **no
hydrologic soil group**, and **no flood labels of any kind**.

⚠️ The `...Rp100` metrics are a **100-metre riparian buffer**, not a 100-year
floodplain. Easy and costly to misread.

**Access**: REST API, verified live —
`https://api.epa.gov/StreamCat/streams/metrics?name=elev,wetindex,precip9120,runoff&areaOfInterest=catchment&state=RI`
returns real rows. Also `StreamCatTools` (R) and `github.com/USEPA/StreamCat`.

**Why this is the unlock**: COMID is the same key CFIM HAND's hydraulic property
tables use. So StreamCat covariates and HAND join directly, with no
geoprocessing, and you inherit a pre-computed TWI rather than deriving one.

---

## 3. EPA RPS Indicator Database v2.8 — trainable in an hour, wrong grain for you

<https://www.epa.gov/rps/data-downloads>

One wide table, **one row per HUC12, 389 columns**, all of the lower 48. I
downloaded Region 4 and confirmed 10,697 rows × 389 populated columns.

It genuinely ships **labels and covariates pre-joined**:
`FLOOD_ZONE_PCT_HUC12` ("% 100-Year Flood Zone in HUC12", straight from FEMA
NFHL, Feb 2021) alongside `SLOPE_MEAN_HUC12`, `ELEVATION_MEAN_HUC12`,
`PRECIP_ANNUAL_HIST_HUC12`, `RUNOFF_ANNUAL_HIST_HUC12`, `IMP_COV_PCT_HUC12`,
`FLOW_ACCUMULATION_MEAN_HUC12`, `STREAMLENGTH_NHD_*`, ~130 NLCD-derived columns.

**Why it is wrong for this project anyway.** The unit is a ~90 km² subwatershed.
`SLOPE_MEAN_HUC12` averaged over 90 km² destroys exactly the local terrain
signal that makes susceptibility modelling work, the label is a continuous area
fraction rather than binary, and — decisively — **a HUC12 contains both Zone A
and Zone AE polygons**, so your domain variable blurs into "fraction of SFHA
that is Zone A." The provenance split stops being a split.

Worth knowing it exists. Not worth using here.

---

## 4. National Structure Inventory — an underrated alternative unit

<https://www.hec.usace.army.mil/confluence/nsi/technicalreferences/latest/technical-documentation>

One point per structure, nationwide, 40+ attributes — including **`firmzone`,
`zone_sub`, `static_bfe`, `grnd_elv_m`**. Bulk GeoPackage/GeoParquet by state.

That is a **ready-made per-point FEMA zone label with the A/AE discriminator
already attached**, at building resolution, nationally. You would still supply
every conditioning factor, but it removes the NFHL polygon-overlay step
entirely and gives you a natural, policy-relevant sampling unit.

Worth a serious look as the alternative to raster-cell sampling — especially
since "which buildings does this affect" is the question flood mapping exists to
answer.

---

## 5. Also checked, also not it

| Source | What it actually is |
|---|---|
| **HydroATLAS / BasinATLAS** | Global, 281 attribute columns per sub-basin, **has slope** (StreamCat doesn't), but global-generic sources (WorldClim/GLC2000/SoilGrids1km vs PRISM/NLCD/STATSGO) and **no flood labels**. `inu_pc` is long-term inundation extent, not hazard. Linke et al. 2019, *Sci Data* 6:283. |
| **USFIMR** | 47 observed flood-extent events (~1996–2017), shapefiles of inundation polygons. **Labels only, zero covariates.** Good for observed-flood validation (RQ4). |
| **FloodCastBench** | **No US coverage** — Pakistan, UK, Australia, Mozambique. Hydrodynamic depth simulation, not susceptibility. |
| **GFSM v1** | 30 m global susceptibility raster, 48.8 GB. Model output, not training data. |
| **Hugging Face / Kaggle** | Nothing usable. One undocumented 6.8 GB GeoTIFF with an empty README; the rest is SAR flood segmentation or synthetic tables. |
| **GitHub** | Every hit is a single-basin study repo. No CONUS assembled dataset. |

---

## 6. Revised assembly plan

**Backbone: StreamCat by COMID at NHDPlus V2 catchment grain.** Catchments are
small enough (2.65 M of them) to keep the A/AE split clean, and the COMID key
buys you elevation, TWI, precipitation, runoff, baseflow index, soil texture and
permeability, NLCD and imperviousness with a table join instead of a raster
extraction.

**Then add, in this order:**

| Still to build | How | Effort |
|---|---|---|
| **Labels + provenance** | NFHL `S_FLD_HAZ_AR`, overlay to catchments, carry `FLD_ZONE` / `STATIC_BFE` | The core work |
| HAND | CFIM HUC6 rasters via TACC — **joins on COMID**, same network | Low |
| Slope, curvature | 3DEP via GEE — StreamCat has no slope | Low |
| Distance to river, drainage density | NHDPlus V2 flowlines, local | Medium |
| SPI | Derived from slope + StreamCat/MERIT contributing area | Trivial |
| Hydrologic soil group | SDA REST, batched by unique mukey | Low |
| Design-storm rainfall | PFDS CSV, or StreamCat `precip9120` as a cheaper proxy | Medium |

That is roughly half the original extraction burden, and it changes the week-2
estimate from "assemble everything" to "join StreamCat, overlay NFHL, derive
five things."

**Provenance note for the writeup:** StreamCat is catchment-aggregated, so your
covariates are catchment means even though the labels are polygon overlays. Say
so. It is a meaningfully different design from per-30 m-cell sampling, and a
reader will want to know which one you did.
