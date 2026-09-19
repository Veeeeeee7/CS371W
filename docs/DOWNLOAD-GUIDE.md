# Download guide

**Updated 2026-09-16.** Week-1 go/no-go passed; Phase 2 replaced with a single
national download after the per-state route proved unusable.

---

## Week-1 result: PASS

Zone counts within the SFHA, all 10 states, from `nfhl_zone_stats.py`:

| state | Zone A | Zone AE | A+AE | A share |
|---|---:|---:|---:|---:|
| IA | 34,890 | 10,607 | 45,497 | **76.7%** |
| FL | 147,124 | 94,396 | 241,520 | **60.9%** |
| LA | 6,318 | 9,042 | 15,360 | 41.1% |
| TX | 19,592 | 48,193 | 67,785 | 28.9% |
| MN | 7,531 | 19,051 | 26,582 | 28.3% |
| AZ | 4,662 | 17,014 | 21,676 | 21.5% |
| MO | 6,989 | 25,688 | 32,677 | 21.4% |
| VT | 816 | 3,159 | 3,975 | 20.5% |
| CO | 2,808 | 20,073 | 22,881 | 12.3% |
| NC | 393 | 56,507 | 56,900 | **0.7%** |
| **TOTAL** | **231,123** | **303,730** | **534,853** | **43.2%** |

Zone A is 43% of mapped SFHA polygons. The domain split has plenty of data on
both sides.

### The discriminator is `FLD_ZONE`, not `STATIC_BFE`

Zone A has a real (non-`-9999`) BFE in **0.0% of polygons in every state**.
Zone AE ranges from **0.4% (CO) to 76.5% (FL)** — a 200× spread reflecting how
each mapping partner populated the attribute, not anything physical. In Zone AE
the elevations normally live as line features in `S_BFE`, since BFE varies along
a reach.

Split on `FLD_ZONE IN ('A','AE')`. Keep `STATIC_BFE` as a candidate in the
alignment battery, not as a feature.

---

## The design change: comparisons must be **within state**

A share ranges from **0.7% (NC) to 76.7% (IA)** — a 100× spread driven by
**state floodplain-mapping policy**, not terrain or urbanization. Some states
ran statewide detailed-study programs; others never did.

This is a third confounder, larger than rural/urban. Pooling A and AE across
states would mostly measure which state a point is in.

1. **All A/AE comparisons are within-state**, then stratified by urbanicity.
   State is a blocking variable, not a domain.
2. **Drop NC from the primary analysis** — 393 Zone A polygons against 56,507
   cannot support a within-state comparison. Keep it as the "detailed-study
   saturated" reference point.
3. **Best contrast states:** FL (147k/94k), TX, LA, MN, MO, IA (inverted but
   large). CO and VT usable with thinner Zone A.
4. **The A-share itself is worth explaining.** Why Iowa 77% and Colorado 12% is
   a legitimate secondary finding that costs nothing to report.

Working set: **FL, TX, LA, IA, MN, MO, AZ, CO, VT**, with **NC** as reference.

---

## Phase 2 — One national download

### Not Hazus

`msc.fema.gov/portal/resources/hazus` distributes **Hazus state databases** —
building inventory and general building stock for loss estimation, derived from
the National Structure Inventory. It is the *exposure* side of risk modelling.
It contains no flood hazard polygons, no `S_FLD_HAZ_AR`, no zone attributes.
Wrong product.

### Not per-state either

The MSC portal requires county and community for a search, and its JSON endpoint
is not exposed in any page JS. Downloading 10 states means hundreds of county
files. Abandoned.

### The national NFHL geodatabase — verified live

```
https://gis.fema.gov/NFHL/NFHL_Key_Layers.gdb.zip
  → 302 → https://hazards.fema.gov/wa/loma/NFHL_Key_Layers_20260303.gdb.zip
  19,590,788,798 bytes (19.59 GB)
  accept-ranges: bytes          <- resumable and parallelizable
  last-modified: 2026-03-05
```

Contains `S_Fld_Haz_Ar`, `S_FIRM_Pan`, `S_BFE`, `S_XS`, `S_Gen_Struct`,
`S_CBRS`, `S_LOMR` plus study-info tables — everything needed, all 50 states,
one file, no clicking.

**The 6-month lag is an advantage here, not a compromise.** State extracts carry
different dates per state (Texas was 2026-09-09, others months older). Since
state is now a blocking variable, a patchwork of extract vintages would confound
directly with it. The national file gives **one consistent snapshot across every
state**, which is what this design wants.

### Download

Parallel and resumable — roughly 4× faster:

```bash
brew install aria2          # if needed
mkdir -p data/raw/nfhl && cd data/raw/nfhl
aria2c -x 8 -s 8 -c --retry-wait=5 -m 30 --console-log-level=warn \
  "https://gis.fema.gov/NFHL/NFHL_Key_Layers.gdb.zip"
```

Plain curl, resumable — rerun the same command after any interruption:

```bash
mkdir -p data/raw/nfhl && cd data/raw/nfhl
curl -L -C - --retry 10 --retry-all-errors --retry-delay 5 \
     -o NFHL_Key_Layers.gdb.zip \
     "https://gis.fema.gov/NFHL/NFHL_Key_Layers.gdb.zip"
```

Verify before extracting:

```bash
ls -l NFHL_Key_Layers.gdb.zip        # expect 19,590,788,798 bytes
unzip -t NFHL_Key_Layers.gdb.zip | tail -3
```

### Extract, subset, reclaim disk

Peak usage is ~60 GB during extraction. Subset immediately to the 10 states and
delete the national GDB — that drops you to a couple of GB.

```bash
unzip NFHL_Key_Layers.gdb.zip
ls -d *.gdb
```

```python
#!/usr/bin/env python3
# subset_nfhl.py -- national GDB -> 10-state GeoParquet
import geopandas as gpd, pyogrio, pandas as pd
from pathlib import Path

GDB = "data/raw/nfhl/NFHL_Key_Layers.gdb"     # confirm exact name from ls
OUT = Path("data/interim"); OUT.mkdir(parents=True, exist_ok=True)

FIPS = {"FL":"12","TX":"48","LA":"22","IA":"19","MN":"27",
        "MO":"29","AZ":"04","CO":"08","VT":"50","NC":"37"}

# layer names are case-sensitive and differ from the per-state GDBs
print([l[0] for l in pyogrio.list_layers(GDB)])

def pull(layer, out_name):
    where = " OR ".join(f"DFIRM_ID LIKE '{f}%'" for f in FIPS.values())
    g = gpd.read_file(GDB, layer=layer, where=where)
    g["STATE"] = g.DFIRM_ID.str[:2].map({v: k for k, v in FIPS.items()})
    g.to_parquet(OUT / out_name)
    print(f"{layer}: {len(g):,} rows -> {out_name}")
    return g

haz  = pull("S_Fld_Haz_Ar", "nfhl_haz_10state.parquet")
pans = pull("S_FIRM_Pan",   "nfhl_firmpan_10state.parquet")

sfha = haz[haz.SFHA_TF == "T"]
print("\ncross-check against the REST go/no-go:")
print(pd.crosstab(sfha.STATE, sfha.FLD_ZONE).loc[:, ["A", "AE"]])
```

The cross-tab at the end should reproduce the table at the top of this file. If
it doesn't, stop — either the filter or the layer is wrong.

```bash
rm -rf data/raw/nfhl/NFHL_Key_Layers.gdb        # after the check passes
```

---

## Phase 3 — NHDPlus V2 (7.8 GB)

```bash
mkdir -p data/raw/nhdplus && cd data/raw/nhdplus
aria2c -x 8 -s 8 -c \
  "https://dmap-data-commons-ow.s3.amazonaws.com/NHDPlusV21/Data/NationalData/NHDPlusV21_NationalData_Seamless_Geodatabase_Lower48_07.7z"
7z x NHDPlusV21_NationalData_Seamless_Geodatabase_Lower48_07.7z
```

V2, not HR — CFIM HAND is burned with V2 flowlines and StreamCat is keyed to V2
COMIDs.

---

## Phase 4 — Boundaries

```bash
mkdir -p data/raw/boundaries && cd data/raw/boundaries
curl -L -O "https://www2.census.gov/geo/tiger/TIGER2024/STATE/tl_2024_us_state.zip"
curl -L -O "https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/tl_2024_us_county.zip"
unzip -o "tl_2024_us_*.zip"
```

County matters — mapping programs are administered at county level, so county is
likely the right unit for the provenance stratum.

---

## Phase 5 — HAND: deferred

MERIT Hydro's 90 m `hnd` via Earth Engine first. Escalate to CFIM's 10 m rasters
only if the pilot shows resolution matters.

---

## Phase 6 — Earth Engine

Do this while downloads run.

```bash
earthengine authenticate
```

```python
import ee
ee.Initialize(project="YOUR_PROJECT_ID")
print(ee.Image("MERIT/Hydro/v1_0_1").bandNames().getInfo())
print(ee.ImageCollection("USGS/3DEP/10m_collection").size().getInfo())
print(ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD").size().getInfo())
```

---

## Layout

```
CS371W/
  docs/
  data/
    raw/
      nfhl/NFHL_Key_Layers.gdb.zip     <- 19.59 GB, deleted after subsetting
      nhdplus/
      boundaries/
    interim/
      nfhl_zone_stats/                 <- done
      nfhl_haz_10state.parquet         <- phase 2 output
      nfhl_firmpan_10state.parquet
    processed/
```

---

## What to send me next

1. `pyogrio.list_layers` output (exact layer names in the national GDB)
2. The `pd.crosstab` cross-check
3. `ls -lh data/interim/*.parquet`

Then I write the point sampler:

- Stratify by **state × county × zone (A/AE) × urbanicity decile**
- Positives inside SFHA, negatives from a fixed distance band outside,
  **identical provenance-blind protocol on both sides**
- Carry `FLD_ZONE`, `ZONE_SUBTY`, `STATIC_BFE`, `DFIRM_ID`, `SOURCE_CIT`, and
  `EFF_DATE` from the `S_FIRM_Pan` join as metadata
- GeoParquet in EPSG:5070
