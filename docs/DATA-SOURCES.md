# Data sources — citations, downloads, gotchas

All URLs verified 2026-09-16 by actually fetching them. Sizes and HTTP statuses
are from live checks.

**Read §9 first if you read nothing else** — four findings change the plan.

---

## 1. FEMA National Flood Hazard Layer (NFHL) — the label

**Cite as a data product. No paper exists.** FEMA's own FGDC citation block:

> Federal Emergency Management Agency, 2015, National Flood Hazard Layer (NFHL),
> Version 1.1.1.0: Washington, D.C., Federal Emergency Management Agency.

Add your extract date — FEMA stamps it into the filename. Use constraints ask
that "Acknowledgement of FEMA would be appreciated in products derived from
these data." Access constraints: none.

### Download

| Scope | URL | Format | Size |
|---|---|---|---|
| **National seamless** | `https://gis.fema.gov/NFHL/NFHL_Key_Layers.gdb.zip` | File GDB | **19.59 GB** |
| Statewide | `https://msc.fema.gov/portal/downloadProduct?productTypeID=NFHL&productSubTypeID=NFHL_STATE_DATA&productID=NFHL_<STFIPS>_<YYYYMMDD>` | **File GDB** | 4 MB (DC) – 2 GB (TX) |
| County | `https://msc.fema.gov/portal/downloadProduct?productTypeID=NFHL&productSubTypeID=NFHL_COUNTY_DATA&productID=NFHL_<CID>` | **Shapefile** | 9–190 MB |

A national file does exist (redirects to a dated
`hazards.fema.gov/wa/loma/NFHL_Key_Layers_YYYYMMDD.gdb.zip`, supports range
requests so it's resumable). It carries only a subset of layers and **lags state
files by ~6 months** — national dated 2026-03-03 vs Texas 2026-09-09.

Product IDs embed a date that rolls monthly, so they must be discovered, not
hardcoded:

```
POST https://msc.fema.gov/portal/advanceSearch
  selstate=<FIPS>
→ JSON: EFFECTIVE.NFHL_STATE_DATA[0].product_NAME
```

### Services

- REST: `https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer` — 32 layers, `maxRecordCount` 2000
- WMS: `https://hazards.fema.gov/arcgis/services/public/NFHLWMS/MapServer/WMSServer` (note **NFHLWMS**, not NFHL — the latter 400s)
- WFS: same path with `/WFSServer`, capped at 1000 features

### Schema — confirmed from live REST and downloaded DBFs

- **Flood zones: `S_FLD_HAZ_AR`, REST layer 28.** Fields: `DFIRM_ID, VERSION_ID,
  FLD_AR_ID, STUDY_TYP, FLD_ZONE, ZONE_SUBTY, SFHA_TF, STATIC_BFE, V_DATUM,
  DEPTH, LEN_UNIT, VELOCITY, VEL_UNIT, AR_REVERT, AR_SUBTRV, BFE_REVERT,
  DEP_REVERT, DUAL_ZONE, SOURCE_CIT`
- **Dates: `S_FIRM_PAN`, REST layer 3.** `EFF_DATE` (effective), `PRE_DATE`
  (preliminary), both true date types.
- Study-level dates in `STUDY_INFO`: `INDX_EFFDT`, `DBREV_DT`. Source dates via
  `L_SOURCE_CIT` (`SRC_DATE`, `PUB_DATE`), joined on `SOURCE_CIT` + `DFIRM_ID`.
- **CRS: EPSG:4269 (NAD83 geographic).** Not 4326, not projected.

### Gotchas

- **`STATIC_BFE` and `DEPTH` use `-9999` as the null sentinel, not NULL.** This
  is the single most dangerous item on this page — the A/AE discriminator is
  "does this polygon have a BFE," and a naive null check silently misclassifies
  every Zone A polygon. Convert `-9999` → NaN on load.
- **`S_FLD_HAZ_AR` carries no date at all.** Vintage requires a spatial join to
  `S_FIRM_PAN`. Many published analyses skip this and treat NFHL as timeless.
- **Use REST layer 0, "NFHL Availability,"** to distinguish *no flood hazard*
  from *not mapped*. Conflating them is the most common substantive error in
  NFHL analyses and would poison the A/AE comparison.
- Filter `SFHA_TF='T'` early. `FLD_ZONE='X'` with `ZONE_SUBTY='0.2 PCT ANNUAL
  CHANCE FLOOD HAZARD'` dominates the polygon count.
- State GDB ≫ county shapefile. County `.dbf`/`.shp` hit the 2 GB ceiling and
  silently truncate; 10 states county-by-county is 800+ downloads.
- `hazards.fema.gov` is flaky — roughly a third of requests died with
  `SSL_ERROR_SYSCALL` and succeeded on retry. Use `--retry 4 --retry-all-errors`.
- Update cadence: monthly.

---

## 2. USGS 3DEP — elevation

**Cite as a data product**; USGS's own format:

> U.S. Geological Survey, 2019, 3D Elevation Program 1-Meter Resolution Digital
> Elevation Model (published 20200606), accessed [date] at
> https://www.usgs.gov/the-national-map-data-delivery

Peer-reviewed reference on data quality: Stoker & Miller (2022), *Remote
Sensing* 14(4):940, `10.3390/rs14040940`.

**Download** — public S3, no credentials:
`https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/`
- 10 m: `13/TIFF/current/<tile>/USGS_13_<tile>.tif` (~475 MB per 1° tile)
- 30 m: `1/` · 1 m: `1m/Projects/<PROJECT>/TIFF/...` (by lidar project, not tile grid)

**API**: TNM Access `https://tnmaccess.nationalmap.gov/api/v1/products` ·
ImageServer `https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer`
(serves derived slope/aspect/hillshade on the fly) · Python `py3dep`

**GEE**: `USGS/3DEP/10m_collection` (band `elevation`). `USGS/3DEP/10m` is
deprecated. No 1 m or 30 m asset.

**Gotchas**: 1 m coverage is partial and project-based, vintages span 2014–2023
even within one state. 10 m tiles are in geographic degrees — **computing slope
or curvature without reprojecting to a metric CRS is wrong, and the error grows
with latitude.** License: public domain.

---

## 3. HAND — Height Above Nearest Drainage

### Which paper for what

| Purpose | Citation |
|---|---|
| Original concept | Rennó et al. (2008), *RSE* 112(9):3469–3481, `10.1016/j.rse.2008.03.018` |
| **Method — cite this by default** | Nobre et al. (2011), *J. Hydrol.* 404(1–2):13–29, `10.1016/j.jhydrol.2011.03.051` |
| Inundation extent | Nobre et al. (2016), *Hydrol. Process.* 30(2):320–333, `10.1002/hyp.10581` |
| **CONUS HAND / CFIM — cite if you use the raster** | Liu, Maidment, Tarboton, Zheng & Wang (2018), *JAWRA* 54(4):770–784, `10.1111/1752-1688.12660` |
| NWM branch-based HAND | Aristizabal et al. (2023), *WRR* 59(5):e2022WR032039, `10.1029/2022WR032039` |

Dataset DOIs: v0.2 `10.13139/ORNLNCCS/1608331` · v0.2.1 `10.13139/ORNLNCCS/1630903`

### Download — CONUS HAND is genuinely available

The CFIM page (`https://cfim.ornl.gov/data/`) says download is "temporarily
unavailable." **That notice is wrong.** The TACC mirror serves the full product:

`https://web.corral.tacc.utexas.edu/nfiedata/HAND/` — 331 HUC6 directories.

Per HUC6: `<huc>hand.tif` (~1.3 GB), plus `fel`, `slp`, `ang`, `p`, `sd8`, `dd`,
`ssa`, `src`, `catchmask`, `waterbodymask`, and
`hydroprop-fulltable-<huc>.csv` (~210 MB). CONUS hydraulic property tables at
`https://web.corral.tacc.utexas.edu/nfiedata/hydraulic-property-table/`.

**Version caveat**: mirror files carry Last-Modified dates of 2016–2017, i.e.
the original NFIE production rather than the 2020 releases holding the DOIs.
State that you used the TACC mirror and give file dates.

**Global 30 m HAND fallback**: `https://glo-30-hand.s3.amazonaws.com/` (verified,
`v1/2021/Copernicus_DSM_COG_10_<lat>_00_<lon>_00_HAND.tif`).
**MERIT Hydro `hnd` band at 90 m is in GEE** — see §5.

**NOAA-OWP FIM**: `s3://noaa-nws-owp-fim/hand_fim/` is **Requester Pays** —
credentials required, you pay egress, ~400 GB inputs. Different product from
CFIM (branch-based HAND over NWM hydrofabric). Do not mix them silently.

**Gotchas**: CFIM HAND is burned with NHDPlus **V2 medium-resolution** flowlines,
so its stream definition is MR — pairing it with HR-derived distance-to-river is
internally inconsistent. ~1.3 GB/HUC6 means hundreds of GB for CONUS. HAND is
undefined over waterbodies; use the supplied masks. License: CC BY 4.0.

---

## 4. NHDPlus — hydrography

**NHDPlus HR production is halted** (USGS moved to 3DHP). Cite:

> U.S. Geological Survey, 20220707, USGS National Hydrography Dataset Plus High
> Resolution National Release 1 FileGDB, https://doi.org/10.5066/P9WFOBQI

**Download HR**: `https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHDPlusHR/National/GDB/NHDPlus_H_National_Release_2_GDB.zip` (**31.3 GB**)

**Download V2** (often the better choice):
`https://dmap-data-commons-ow.s3.amazonaws.com/NHDPlusV21/Data/NationalData/NHDPlusV21_NationalData_Seamless_Geodatabase_Lower48_07.7z` (**7.8 GB**)

**Prefer V2 here**: it's a quarter the size, its COMIDs join directly to CFIM
HAND's hydraulic property tables, and its VAA table gives reach slope and
cumulative drainage area for free.

**API**: `https://hydro.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/MapServer` ·
NLDI `https://api.water.usgs.gov/nldi/linked-data` · Python `pynhd` · R `nhdplusTools`

**GEE**: none. No NHD asset exists. Upload flowlines as a table asset or compute
distance locally.

**Gotchas**: the old canonical host `horizon-systems.com/NHDPlus/V2NationalData.php`
**returns HTTP 523** — many papers still link there. HR and V2 use incompatible
IDs (NHDPlusID vs COMID, no clean crosswalk) and have very different stream
densities, which changes every distance and density metric. **Pick one, up front,
and say which.**

---

## 5. MERIT Hydro — the GEE shortcut

> Yamazaki, D., Ikeshima, D., Sosa, J., Bates, P.D., Allen, G.H., & Pavelsky,
> T.M. (2019). MERIT Hydro: A high-resolution global hydrography map based on
> latest topography datasets. *WRR* 55:5053–5073. `10.1029/2019WR024873`

**GEE: `MERIT/Hydro/v1_0_1`** — bands `elv, dir, wth, wat, upa, upg, hnd,
viswth` at 92.77 m.

This matters more than it looks. GEE has **no flow-accumulation primitive**, so
without MERIT you'd need TauDEM offline. MERIT's `upa` (upstream drainage area)
makes TWI and SPI one-line GEE computations, and `hnd` gives you a 90 m HAND for
free. Direct download is registration-gated via Google Form → Dropbox; use GEE.

License: dual CC-BY-NC-4.0 / ODbL-1.0. **The non-commercial arm is a real
constraint** — check which applies to your use.

---

## 6. NLCD — land cover and imperviousness

**Current release: Annual NLCD Collection 1, v1.2, published 30 June 2026,
covering 1985–2025, CONUS only, 30 m.** The old epoch-based NLCD
(2001/2006/…/2021) is superseded.

**Cite the methods paper** (listed by USGS as the reference for this release):

> Fleckenstein, R., Wellington, D., Jin, S., Tollerud, H., Brown, J.F., Dewitz,
> J., Pastick, N.J., Barber, C.P., O'Brien, A., Spanier, M., 2026. A framework
> for integrating spatiotemporal deep learning methods with Landsat for annual
> land cover and impervious surface mapping. *Remote Sensing of Environment*
> 338, 115347. `10.1016/j.rse.2026.115347`

Plus the data release: USGS, 2024, Annual NLCD Collection 1 Science Products
(ver. 1.2, June 2026), `10.5066/P94UXNTS`.

**Do not mix up the older citations**: Homer et al. 2020 → NLCD 2016 era.
Jin et al. 2023 → NLCD 2019 era. "Dewitz" entries are data releases, not papers.

**Download**: ScienceBase, one zip per year per product —
Land Cover `https://www.sciencebase.gov/catalog/item/697b9279b66b0197c3043cc3` ·
Fractional Impervious `https://www.sciencebase.gov/catalog/item/697b907eb66b0197c3043c9f`
Enumerate file URLs from `?format=json`. Interactive clip: `https://www.mrlc.gov/viewer/`

**GEE**: Annual NLCD is **not in the official catalog**. Options:
- Official legacy: `USGS/NLCD_RELEASES/2021_REL/NLCD` — bands `landcover`,
  `impervious`, `impervious_descriptor`; epochs through 2021
- Community: `projects/sat-io/open-datasets/USGS/ANNUAL_NLCD/LANDCOVER` and
  `.../FRACTIONAL_IMPERVIOUS_SURFACE`

**Gotchas**: CONUS only — no AK/HI/PR in Collection 1. Version suffix is baked
into filenames (`_C1V0` vs `_C1V2`); mixing versions across years silently breaks
a time series. Legacy NLCD and Annual NLCD are not year-to-year comparable.

---

## 7. NOAA Atlas 14 — design-storm rainfall

**Atlas 14 is still authoritative. Atlas 15 has not superseded it** — preliminary
CONUS estimates due Sept 2026, publication 2027.

**Cite the volume you used** (no paper; the volumes are the product):

> Perica, S., et al., [year], Precipitation-Frequency Atlas of the United States,
> Volume [N], Version [V]: [Region]. NOAA Atlas 14, NOAA, National Weather
> Service, Silver Spring, MD.

Author roster differs per volume — pull from the PDF title page. Volume table:
`https://www.weather.gov/owp/hdsc_currentpf`

**The point API is the useful thing** — verified working:

```
https://hdsc.nws.noaa.gov/cgi-bin/new/fe_text_mean.csv?lat=<LAT>&lon=<LON>&type=pf&data=depth&units=english&series=pds
```

Returns the full 19-duration × 10-ARI table plus a header naming the volume and
version. **Carry that header into your table as a provenance column.** Rate-limit
yourself — this is a CGI script, not a CDN.

**Bulk grids**: `https://hdsc.nws.noaa.gov/pub/hdsc/data/<region>/<region><ARI>yr<dur>a.zip`
(`a` = mean, `al`/`au` = CI bounds). Region codes: `sw` (Vols 1 *and* 6),
`orb`, `pr`, `hi`, `ak`, `mw`, `se`, `ne`, `tx`, `inw`.

**GEE**: none.

### Gotchas — this is the painful dataset

- **No seamless national grid exists.** `pub/hdsc/data/usa/` is empty. You
  mosaic volumes yourself and accept seams, which worsen at rarer return
  intervals — exactly the ones design storms use.
- **Volumes span 2004–2024 in publication date.** Two points 50 km apart across a
  volume boundary can rest on rainfall records two decades apart. Not fixable,
  only documentable.
- **Oregon and Washington have no Atlas 14 coverage at all.** Verified: a PFDS
  query at Portland returns "Selected location is not within a project area."
  They fall back to NOAA Atlas 2 (1973), which offers only 2-yr and 100-yr at
  6-hr and 24-hr. See §9.
- Atlas 15 lands in 2027 — isolate this layer behind a swappable interface.

---

## 8. Soils — hydrologic soil group

**Use gNATSGO.** SSURGO has holes and ships per-survey-area (~3,000 downloads);
gSSURGO has the same holes; gNATSGO gap-fills with STATSGO2 and is 100 % seamless.

> Soil Survey Staff (2026). Gridded national soil survey geographic (gNATSGO)
> database. USDA Natural Resources Conservation Service.
> https://nrcs.app.box.com/v/soils

**The fast path is Soil Data Access REST**, verified returning `hydgrp` directly:

```
POST https://sdmdataaccess.nrcs.usda.gov/Tabular/post.rest
{"format":"JSON+COLUMNNAME","query":
 "SELECT mu.mukey, c.compname, c.comppct_r, c.hydgrp
  FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('point(-97.74 30.27)') AS m
  INNER JOIN mapunit AS mu ON mu.mukey = m.mukey
  INNER JOIN component AS c ON c.mukey = mu.mukey
  ORDER BY c.comppct_r DESC"}
```

At scale: sample `projects/sat-io/open-datasets/gNATSGO/raster/mukey` in GEE
alongside your other layers, then run **one** SDA query per batch of unique
mukeys and join locally. Far fewer unique mukeys than points.

### Gotchas

- **`hydgrp` is not in GEE.** The mukey raster is; the attribute is not. This is
  the biggest trap in the assembly.
- **`hydgrp` lives on `component`, not `mapunit`** — one mukey has many
  components with different groups. You must pick an aggregation rule (dominant
  condition by `comppct_r` is conventional) and document it.
- **Dual groups**: values include `A/D`, `B/D`, `C/D` for soils whose group
  depends on drainage. Naive letter parsing produces garbage.
- Urban land, water and misc areas often have `hydgrp = NULL`. Plan a fallback.
- `pygeohydro.soil_gnatsgo` hits the Planetary Computer mirror and **will not
  give you hydgrp either.**

---

## 9. Four findings that change the plan

**1. Drop Washington (and any Oregon plan) from the state list.** Neither has
NOAA Atlas 14 coverage. Their design-storm layer would be 1973 Atlas 2 data with
four duration/ARI combinations total — a qualitatively different variable sitting
in the same column. Replace with e.g. **Missouri or Tennessee**. Revised list:
FL, LA, IA, CO, AZ, NC, VT, TX, MN, MO.

**2. `STATIC_BFE` nulls are `-9999`, not NULL.** The A/AE discriminator depends
entirely on this. Convert on load, and assert that the `-9999` count matches the
Zone A polygon count as a sanity check.

**3. CONUS HAND is downloadable** despite CFIM saying otherwise — the TACC mirror
works. But MERIT's 90 m `hnd` in GEE may be good enough for a first pass and
costs nothing. Start with MERIT; escalate to the 10 m CFIM rasters only if the
pilot shows HAND resolution matters.

**4. Decide NHDPlus V2 vs HR before writing any code.** Distance-to-river and
drainage density are not comparable across the two networks, and CFIM HAND is
built on V2. Recommendation: **V2 throughout**, for internal consistency with
HAND and a quarter the download.

---

## 10. Assembly path

**GEE handles six of eight covariates in one `sampleRegions` call:**
elevation, slope, aspect, curvature (`convolve` with 3×3 kernels — no built-in),
`upa` → TWI and SPI, `hnd`, land cover, imperviousness, and `mukey`.

**Three things GEE cannot do**, all local:

| Covariate | Why | How |
|---|---|---|
| Distance to river | No NHD asset in GEE | GeoPandas `sjoin_nearest` in EPSG:5070 |
| Drainage density | Needs aggregation over HUC12 | dissolve flowline length ÷ area |
| Hydrologic soil group | `hydgrp` not in GEE | SDA REST, batched by unique mukey |
| Design-storm rainfall | Atlas 14 not in GEE | PFDS CSV per point, or per-volume grids |

**Reproject to EPSG:5070 (CONUS Albers) before any slope, curvature, distance or
area computation.** Computing slope on 1/3 arc-sec degree cells is the most
common error in this pipeline and it worsens northward across a 10-state span.
