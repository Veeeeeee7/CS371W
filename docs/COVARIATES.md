# Covariates: what goes in the model, what never does

Status: decided 2026-09-18, after the Earth Engine smoke test returned values for
every band and two of them turned out to be artefacts.

---

## 1. The rule this document exists to enforce

Every column in `data/interim/points/points_{ST}.parquet` is in exactly one of two
sets.

**Features** — things a model may see. They describe the *place*.

**Metadata (`meta_` prefix)** — things that describe *how FEMA made the map*. A
model that sees any of these is predicting its own label.

The prefix is the enforcement mechanism: the training code selects features by
excluding `meta_*` and the identifier columns, so adding a leaky column by
accident requires actively naming it.

### Why this is not paranoia

Measured across the ten states, `STUDY_TYP` alone predicts Zone A vs Zone AE at
**68.8%** against a **56.4%** base rate, and several of its levels are effectively
deterministic:

| STUDY_TYP | Zone A | Zone AE | % A |
|---|---:|---:|---:|
| SFHA WITH UNPUBLISHED BFE | 8,070 | 8 | 99.9 |
| SFHAS WITH LOW FLOOD RISK | 42,580 | 153 | 99.6 |
| SFHAS WITH MEDIUM FLOOD RISK | 10,728 | 34 | 99.7 |
| SFHA WITHOUT BFE | 2,777 | 6 | 99.8 |
| SFHA WITH BFE AND FLOODWAY | 42 | 14,105 | 0.3 |
| SFHA WITH BFE NO FLOODWAY | 324 | 5,336 | 5.7 |
| REDELINEATION | 687 | 6,810 | 9.2 |
| NP (not populated) | 153,005 | 190,818 | 44.5 |

`meta_panel_scale` is the same hazard and is sneakier, because it is a plain
number that looks like terrain. Measured on the Vermont point set, where
`STUDY_TYP` is 100% unpopulated and therefore carries nothing at all:

| domain | median map scale |
|---|---:|
| A | 1:12,000 |
| AE | 1:6,000 |

So even in a state with no study metadata, the panel's map scale separates the two
domains. That is a genuinely useful *explanatory* variable — it is a continuous
measure of study detail, which is what the whole project is about — and a fatal
*predictive* one.

`meta_dist_to_sfha_m` is the third: it is the distance to the thing being
predicted. Recorded for diagnostics, never a feature.

---

## 2. Features

### 2.1 Point-level terrain — USGS 3DEP, 10 m, via Earth Engine

| feature | notes |
|---|---|
| `elev` | metres |
| `slope` | degrees, `ee.Terrain.slope` |
| `aspect` | degrees; use sin/cos, not the raw value |
| `curvature` | Laplacian of the DEM; no tuning constants |

`.mosaic()` strips the projection and `ee.Terrain` silently returns null on an
image without one, so the DEM must be given its native projection back with
`setDefaultProjection()` before any terrain call. Sampling is at `scale=10`.

### 2.2 Point-level hydrology — MERIT Hydro, ~90 m

| feature | notes |
|---|---|
| `HAND` | height above nearest drainage, metres. The strongest single predictor in the flood-susceptibility literature, and it involves no division by a floored slope. |
| `log1p_upa` | log upstream area. Labelled a **channel-proximity proxy**, not a contributing area — see §4. |

### 2.3 Catchment context — EPA StreamCat, per NHDPlusV2 catchment

Joined by COMID after a spatial join of points to NHDPlusV2 catchments.

| feature | metric name | notes |
|---|---|---|
| wetness index | `WetIndexCat`, `WetIndexWs` | Mean Composite Topographic Index over the catchment (`Cat`) and the full upstream watershed (`Ws`). **Constant across a ~1–3 km² catchment**, so it is catchment context, not a point TWI. It must be described that way in the write-up. |
| base flow index | `BFICat`, `BFIWs` | % of streamflow from groundwater |
| runoff | `RunoffCat`, `RunoffWs` | mm |
| precipitation | `Precip8110Cat`, `Precip8110Ws` | PRISM 30-year normal, mm |
| soils | `ClayCat`, `SandCat`, `PermCat`, `KffactCat` | STATSGO |
| geology | `HydrlCondCat` | lithological hydraulic conductivity |
| elevation | `ElevCat`, `ElevWs` | for consistency checks against 3DEP |

API: `https://api.epa.gov/StreamCat/streams/metrics?name=<metric>&comid=<list>`.
Full metric dictionary at `https://api.epa.gov/StreamCat/streams/variable_info`.

Cite: Hill, R.A. et al. (2016), *The Stream-Catchment (StreamCat) Dataset*, JAWRA
52(1):120–128.

### 2.4 Land surface — NLCD 2021, 30 m

| feature | notes |
|---|---|
| `landcover` | categorical, one-hot or target-encode |
| `impervious` | % — also the urbanicity stratifier |

### 2.5 Still to assemble

| feature | source | status |
|---|---|---|
| distance to nearest flowline | NHDPlus V2 flowlines | **needed** — this is the honest version of what `upa` is standing in for |
| drainage density | NHDPlus V2 flowlines / catchment area | needed |
| hydrologic soil group | gNATSGO via SDA REST, or `projects/sat-io/...` in GEE | optional; StreamCat soils may suffice |

---

## 3. Metadata — never features

| column | what it encodes |
|---|---|
| `meta_fld_zone` | the domain label itself |
| `meta_zone_subty` | floodway, coastal, etc. |
| `meta_study_typ` | see §1 — predicts the domain at 68.8% |
| `meta_study_regime` | collapsed mechanism: NOT_POPULATED / BLE_RISKCLASS / REDELINEATION / BFE_DETAIL / NO_BFE / DIGITAL_CONVERSION |
| `meta_static_bfe`, `meta_depth` | −9999 sentinels already cleaned to NaN |
| `meta_source_cit`, `meta_fld_ar_id` | study citation, polygon id |
| `meta_poly_area_km2` | source polygon size |
| `meta_dist_to_sfha_m` | distance to the label |
| `meta_eff_date` | panel effective date |
| `meta_panel_scale` | map scale — see §1 |
| `meta_panel_typ` | printed / not printed, countywide / community |
| `meta_bfe_km_per_km2` | BFE line density per DFIRM, the continuous study-detail measure |

These are the **explanatory** variables. The transfer gap gets regressed against
them; the model never sees them.

---

## 4. Why TWI and SPI were dropped

The Earth Engine stack initially carried TWI and SPI. Both returned values at all
three test points, which is what made them dangerous.

**The slope floor decides the answer.** TWI = ln(a / tan β) diverges as slope goes
to zero, so every implementation floors tan β; ours used 0.001. At the Winooski
valley-bottom point:

```
slope       = 0.035 degrees
tan(slope)  = 0.000611      <-- the real value
used        = 0.001000      <-- the floor
TWI         = 11.08         <-- set by the constant, not the terrain
```

The floodplain is where slope is smallest, so the floor binds hardest at exactly
the cells that carry the positive label. A feature whose value is decided by a
preprocessing constant, on the cells that define the positive class, is the trap
Steger et al. (2017) named: the model learns the pipeline, not the landscape.
Sweeping the floor would move the feature and therefore move the result.

**The numerator is sub-cell.** MERIT Hydro's upstream area is 90 m (~8,100 m² per
cell). At the three test points it reads 0.006 / 0.006 / 0.104 km² — **0.74, 0.74
and 12.84 cells**. A 3-cell focal maximum changes it by 1× / 3× / 1×, so this is
not misalignment a nudge would fix; the points genuinely carry near-zero upstream
area in MERIT. At that resolution `upa` behaves as a binary "am I in a mapped
channel" flag, not a continuous contributing area — and it is the input to both
TWI and SPI.

**A 10 m TWI is not available.** Earth Engine has no flow-accumulation algorithm
for an arbitrary DEM, so recomputing from 3DEP is not an option inside this stack.

So the wetness signal is split into pieces that are each defensible on their own:
point-level `HAND`, catchment-level `WetIndexCat`/`WetIndexWs`, and point-level
`slope`/`curvature` with no constants of ours in them. `upa` survives only as
`log1p_upa`, described as channel proximity.

`check_ee.py` reproduces the arithmetic above on every run, so the decision stays
checkable rather than becoming folklore.

---

## 5. Splitting

Every split — train/test, CV fold, OODSelect subset — is made on **`block_id`**
(10 km grid cell), never on individual points. Points within a block share
terrain, and several StreamCat variables are constant across a whole catchment, so
a random point split would put near-duplicates on both sides and reproduce the
>0.95 AUCs the flood-susceptibility literature is criticised for.

Cite: Roberts, D.R. et al. (2017), *Cross-validation strategies for data with
temporal, spatial, hierarchical, or phylogenetic structure*, Ecography 40:913–929.

---

## 6. Order of work

1. ~~Sample points~~ — `sample_points.py`, verified on Vermont.
2. Spatial join points → NHDPlusV2 catchments (COMID). Needs the NHDPlus V2
   download (7.8 GB, still outstanding).
3. Pull StreamCat metrics by COMID.
4. Pull the Earth Engine stack by `lon`/`lat` in batches.
5. Distance to flowline and drainage density from the same NHDPlus download.
6. Assemble `data/processed/dataset_{ST}.parquet`.
