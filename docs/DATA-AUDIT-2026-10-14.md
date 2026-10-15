# Data audit, 2026-10-14: Florida sampling protocol v2

Brief: `docs/TASK-DATA-AUDIT.md`. Plan: `docs/PROJECT-PLAN.md` v4 §2a and §4. Run order:
`docs/RUN-FL-V2.md`. Logs: `logs/2026-10-14/`. Results: `results/2026-10-14/`.

**What happened.** On 2026-10-14 the code for Parts A–D was written and smoke-tested. The same
night (14–15 Oct) the whole routine ran on the real data, steps 1–16, with Victor's choices:
- band negatives kept inside Florida (decision 1, option b);
- the DEM approved up to 10 GB, preferably about 5;
- Earth Engine project `cs371w-gis`.

Every step completed. Open items:
- **Atlas 14 gap:** NOAA's Atlas 14 server (PFDS) failed every request, so 35 of the 1,664 10 km
  blocks (191 points, 0.24%) have no Atlas 14 values (§3, fix in §12).
- **B7, the audit's one WRONG verdict:** two Zone AE positives lie 196 and 214 m outside the
  TIGER state line, beyond the audit's 100 m tolerance. Accepted as a FEMA-versus-Census line
  difference over water (§5).

Before the run, Victor narrowed the project to Florida and deleted the other nine states' files
(§13).

Verdicts: ✓ correct · ✗ missing or wrong · ⚠ works but needs a decision.

---

## 1. Archive (A0)

`archive_sampling_v1.sh` moved (plain `mv`, same volume) every Florida file the rebuild
overwrites, or that is keyed by v1 `point_id`, into `data/interim/archive/sampling-v1/`, keeping
relative paths. v2 reissues `FL0000000…FL0079999` for different locations, and every pull script
treats an existing output as done, so a v1 file left in place would have been joined onto v2
points silently. Ran 2026-10-14: 15 moved, 2 copied, 264 MB in the archive; log
`logs/2026-10-14/A0_archive.log`, list in `MANIFEST.txt`.

| moved | size | why |
|---|---|---|
| `data/interim/points/points_FL.parquet` | 5.7 MB | overwritten by the v2 build |
| `data/processed/dataset_FL.parquet` | 23 MB | overwritten by `pull_gee.py --merge FL` (git-tracked: shows as modified) |
| `data/interim/gee/FL/` (50 shards) | 6.9 MB | shards count as done by file name |
| `data/interim/hydro/comid_FL.parquet`, `river_FL.parquet` | 1.0 + 3.2 MB | skipped if present, joined on `point_id`; `river_FL` has no geometry |
| `data/interim/aotil/cells/FL_band_{AE,A}_s0.npz` and `_models.csv` | 49 MB | `zoo.py` skips an existing cell |
| `data/interim/aef/aef_FL_{2017,2024}.parquet`, `aef/{2017,2024}/FL/`, `aef_tiles_FL_*.geojson` | 174 MB | keyed by `point_id`; `aef_checks.py` joins them |

Copied, not moved, because the run updates them in place: `data/interim/atlas14/blocks.parquet`
and `data/processed/FEATURES.txt`. Left in place: `streamcat_FL` (keyed by COMID), `pan_FL`,
`s_bfe_FL`, `s_lomr_FL`, `idx_FL`, `haz_FL_5070`, `aotil/cells-laptop/` (a stale 24-model pilot;
`run_aotil_laptop.sh` clears it before rebuilding). `results/2026-09-*` stay as the v1 results.

v1 Florida, recorded before the move (`logs/2026-10-14/A0_audit_v1.log`): 98,578 points,
19,963 positives, 11,326 band and 18,000 shadedX negatives per zone; band subset 62,578;
0 pairs under 100 m. The v1 points file has no `neg_rank` column, so it was not built by the
code in HEAD; PLAN's "v1: up to 2:1" is therefore unverified and is not repeated here.

## 2. Protocol v2 as built (A1, A2)

```
sample_points.py FL --protocols band --exact --ring-oversample 6 --state-polygons data/interim/gis/counties_FL.parquet
```

23 minutes. Log `A2_sample_points_FL.log`, audit `A2_audit_FL.log`.

| setting | value | achieved | verdict |
|---|---|---|---|
| spacing, any two points | `min_sep_m` 100 (was 200 within a stratum) | 0 pairs under 100 m; minimum nearest-neighbour distance 100.114 m in EPSG:5070; median 431 m | ✓ |
| spacing across classes and zones | one whole-state pass at the full 100 m (was 50%) | in the line above; the pass removed 159 A and 179 AE band negatives and 2 A positives | ✓ |
| positives per zone | exactly 20,000 | 20,000 / 20,000; drawn with a 3% surplus, cut in random order after the pass, never to reach a ratio | ✓ |
| band negatives per positive | exactly 1:1 | 20,000 / 20,000. Ring candidates 370,063 (A) and 370,790 (AE); 89% and 85% fail the 500 m rule; after the panel and state masks and thinning, 32,998 and 48,212 remain against 20,600 needed. Surplus trimmed by `meta_neg_rank` | ✓ |
| band negatives inside Florida | **new, decision 1 (b)**: `--state-polygons` with the 67 TIGER/Line 2026 Florida counties | the mask removed 2,609 (A) and 1,092 (AE) candidates; 0 band negatives outside Florida. Without it (scratch build, no log kept), 1,761 (4.4%) fell in Alabama and Georgia, inside Florida panel sheets that cross the state line, where the 500 m rule saw no floodplain map. v1 had 1,038 (4.6%) | ✓ |
| shadedX | not built | `--protocols band` | ✓ |
| Florida total | 80,000 | 80,000 | ✓ |
| area-proportional draw, 2% cap | unchanged (the cap applies to candidates: 1,200 of 60,000) | largest single polygon 2.23% (A) / 2.09% (AE) of a zone's positives; top 10 polygons 20.8% / 16.2%; positives from 592 (A) and 950 (AE) polygons | ✓ |
| band 500–5,000 m from any SFHA | unchanged | 500–4,965 m; recomputed for all 40,000 by `audit_dataset.py` B16 (within 1 m) and by `pull_point_extras.py` (exact) | ✓ |
| inside a FIRM panel | unchanged (any DFIRM's panel) | 40,000 / 40,000 | ✓ |
| 10 km blocks, 50 km tiles, seed | unchanged | 1,664 blocks, 102 tiles; seed 20260918 | ✓ |
| ring polygons | `MAX_RINGS` 1,500 not binding | 592 (A), 950 (AE) | ✓ |
| A/AE balance | exact | identical strata | ✓ |

Smoke-test observations (scratchpad, no log kept in `logs/`): two scratch builds without the mask
were identical in every column, and the real build with the full-resolution TIGER counties gave
exactly the counts of a scratch build that used TIGERweb counties.

Other changes in `sample_points.py`, with no effect on the sampled points:
- New per-point columns `meta_src_row_id` (the polygon that labelled a positive, or seeded a
  band negative's ring) and `meta_neg_rank` (`neg_rank` renamed under the `meta_` rule).
- The build config is stored in the parquet metadata.
- `--exact` writes nothing and exits 1 when any stratum is short, and names the remedy.
- `enforce_ratio()`, which trimmed positives, is deleted.
- `attach_panel_meta` takes the point's own DFIRM's panel first, then the lowest FIRM_PAN; the
  old first-match rule gave 19% of v1 points another county's `meta_eff_date`.
- `--audit` tests every pair exactly.
- Docstring notes 2, 3, 4 and 8 restate the v2 values and reasons.

## 3. Covariates (A3)

| step | result |
|---|---|
| Earth Engine `pull_gee.py FL --project cs371w-gis` | 40/40 chunks in 2.4 min, 0 failures. JRC Global Surface Water **v1.4**: checked live under `cs371w-gis` before the pull (`pull_gee.gsw_asset()` returns `JRC/GSW1_4/GlobalSurfaceWater`; 1.3 is deprecated), not written to the pull log. Worst band coverage 91.9%: MERIT `hand`, `log10_upa`, `tpi_broad` are missing for 8.1% of points, 99% of them open-water band negatives |
| `pull_gee.py --merge FL` | 80,000 rows; `FEATURES.txt` left as is. The merge now refuses shards from another sampling build (shards carry `pt_lon`/`pt_lat`) |
| Atlas 14 `pull_atlas14.py FL` | ⚠ The pull (about 23:07–23:09 EDT) failed all 35 new-block requests; its log records failures, not status codes. Manual checks from 23:12 to 00:14, including the selftest's Vermont point and an hour-long poll, all got **HTTP 503 Service Unavailable**. 1,629 of 1,664 blocks come from the cache (at v1 block-mean positions). **191 points (0.24%) have NaN** `p100_24hr`, `p100_60min`, `p100_4day`, `p2_24hr`, `growth_100_2`: 7 A negatives, 176 AE negatives, 8 AE positives. 83% of them are on open water and 24 of the 35 blocks hold only water points (many in the Keys, Florida Bay and the Gulf), so some may keep failing as outside Atlas 14's area even when PFDS is back (v1 had one such block). hgb/xgb handle NaN; the other families impute the median. Fix in §12 |
| NHDPlus V2.1 (03N/03S/03W, 547 MB download; 2.0 GB on disk with the extracted units) | `prep_nhdplus_units.py` built `NHDPlusV21_FL.gdb` (51,501 catchments, 52,351 network flowlines). In the 2026-10-14 smoke test (no log kept) it reproduced v1's national-file hydro values on the 98,578 v1 points exactly (distances within 7.4e-5 m); the COMID part was re-confirmed by the document check |
| `pull_hydro.py --catchments/--flowlines FL` | catchment found for 90.7% of points (11,034 distinct COMIDs; v1 band subset 92.8%): NHDPlus has no catchments over open water. 37,768 river and 4,472 coastline flowlines; coast distance 100% filled |
| StreamCat | cache reused (queried per county, 2026-09-19; v2 covers the same 67 counties). Every COMID has a StreamCat row; 12 points lack `bfiws` (v1 band subset: 377) and 763 lack at least one StreamCat column, mostly the cat-level soil columns |
| `pull_hydro.py --merge FL` | 27 hydro features, 7 meta columns; FEATURES.txt 45 → 45 |
| `pull_climate.py`, `pull_extras.py` | produce none of the 45 features: not run |
| Florida-only checks (step 6) | `FEATURES.txt` byte-identical to v1 (sha256 `31b8654f…8301`); the non-FL Atlas 14 cache rows unchanged |

## 4. v2 against v1 (A4) — `results/2026-10-14/data_audit/audit.md`

| check | result | verdict |
|---|---|---|
| columns | the same 45 features and metadata columns as v1, plus `meta_src_row_id` and `meta_neg_rank`; no shadedX rows; `point_id` unique | ✓ |
| medians and missingness by domain × label, band subset | 180 cells; 24 flagged (7 by median, 20 by missingness). Largest median move 0.077 of v1's IQR (`p100_60min`, AE, 109 → 108 mm); largest missingness change 2.25 points (A band negatives' StreamCat soil columns, 6.2% → 4.0%). No feature moved materially | ✓ (audit: DECIDE, none material) |
| close pairs and MERIT | 43 of the 1,179 pairs closer than 125 m have identical `hand` and `log10_upa` (3.6%, 86 points), against 2.1% at 125–250 m (v1: 3 of 54). At most 27 of the 43 also match on raw upstream area, so only a minority can be two points in one ~90 m MERIT cell; identical values do not prove a shared cell | ⚠ disclose |

## 5. Part B: the point table — `results/2026-10-14/data_audit/audit.md`

| item | result | verdict |
|---|---|---|
| counts by domain × label × protocol | 20,000 × 4 | ✓ |
| seed-0 50 km tile split (`zoo.split_tiles` + `zoo.subset`) | train 28,109 · ID test 10,235 · OOD test 12,771; 71 train / 24 test tiles of 102 (31 drawn for test; 7 without positives of both zones are dropped, 1,663 points). v1: 22,658 · 7,634 · 9,412; 71 / 23 | ✓ (audit: DECIDE, for the share below) |
| positive share, ID vs OOD halves | seed 0: 0.403 vs 0.476 (gap +0.074; v1 +0.105). The gap is a property of the split: seeds 1 and 2 reverse it on v2 (−0.045, −0.092), and over 20 split seeds it averages about 0 in both v1 and v2 (−0.008, −0.015; sd ≈ 0.05; mean \|gap\| 0.040 vs 0.044). So 1:1 did not narrow it. Report O1 in AUC | ⚠ |
| how a band negative gets its zone | text below. B15: every band negative's ring polygon has its zone (0 exceptions). B16: 15.0% of Zone A band negatives lie nearest a Zone AE polygon; 28.9% of Zone AE negatives lie nearest Zone A (plus 34.1% nearest AH/AO/VE) | ✓ |
| feature provenance | `docs/FEATURE-SOURCES.md`: 45 features from seven sources (3DEP 6, MERIT 3, NLCD 2, JRC GSW 2, NHDPlus 7, StreamCat 20, Atlas 14 5). StreamCat's 20 summarise other products; say so in Methods | ✓ |
| missingness | 34 of 45 features have gaps (32 above 0.01%; `landcover` and `impervious` miss one offshore point). aspect 61.9% (NaN on flat ground, by design). **Label-correlated:** 29.7% of AE band negatives lack HAND against 0.1% of AE positives; StreamCat about 31% vs 2.5%. Cause: 34.8% of AE band negatives are open water (AE positives 15.8%): the 5 km rings reach bays inside the panels. hgb and xgb (75 of 300 models) see the NaN and can learn "missing = negative"; the other families see the imputed median | ⚠ decision 2 |
| constant / near-constant features | none | ✓ |
| FEATURES.txt hygiene | 45 entries, all present; no `meta_*`, lon, lat, elevation, id, block or tile column | ✓ |
| lon/lat CRS and inside Florida | lon/lat are NAD83 (EPSG:4269), not EPSG:4326 as the brief says; about 1 m apart here. All band negatives inside the Florida counties. 4 Zone AE positives lie outside the TIGER line, where FEMA's polygons reach past it in the water: 3 in Perdido Bay (Escambia: 61, 196 and 214 m) and 1 on the St. Marys (Baker, 6 m). The audit says WRONG because two are beyond its 100 m tolerance. Accepted as a line difference; positives are never trimmed | ✗ accepted |
| `meta_fld_ar_id` → polygon | all 40,000 positives resolve through `meta_src_row_id` to a polygon of their zone that contains them | ✓ |
| `meta_fld_zone` vs `domain`; `county_fips` | consistent; 0 null, all `12…` | ✓ |
| `meta_eff_date` | 55 null, all band negatives whose chosen panel has no EFF_DATE; the own-DFIRM panel rule reproduces for every point; 2,309 points lie only in neighbouring DFIRMs' panels | ✓ (audit: DECIDE, nulls only) |
| `channel_slope` no-data | NHDPlus's −9998 code enters as a value for 43 points (v1 band subset: 30; 61 with shadedX) | ⚠ decision 3 |

**Methods text: how a band negative gets its zone** (confirmed against the code):
> Band negatives are drawn separately for each zone. We buffer the Zone A (or Zone AE) polygons
> that supplied that zone's positives by 5 km and draw candidate points inside
> those buffers, choosing a buffer in proportion to its area (so overlapping buffers are sampled
> more densely). A candidate is kept only if it lies inside a FIRM panel and inside Florida, and
> its distance to the nearest Special Flood Hazard Area polygon of any zone is between 500 m and
> 5 km. A negative takes the zone of the polygons whose buffer produced it, not the zone of its
> nearest floodplain, so a "Zone A" negative can lie closer to a Zone AE polygon than to any
> Zone A polygon.

## 6. Part C: models, selections and regions

**C1–C2. Cell `FL_band_AE_s0`.** 300 models in 22 s. `aotil/audit_cell.py`: **PASS**, every check.
- Shapes: S_ood 300 × 12,771, S_id 300 × 10,235.
- Families: dt 38, et 38, hgb 38, lr 38, mlp 37, nb 37, rf 37, xgb 37.
- Every column joins to its dataset row through `point_id`, for both halves.
- Both halves use the same 24 test tiles, and every tile holds both zones.
- 10 refits, including xgb, mlp and hgb, match with max |diff| 0.0.
- `zoo_specs`, `split_tiles`, `subset` and `build` are AST-identical to HEAD.
- AUC is now ranked on the raw p. A bug already in HEAD ranked it on 1−(1−p), which broke ties:
  coarse dt models moved by up to 0.095, while the aggregate barely moved.

| quantity (300 models) | v2 | v1 (corrected AUC) |
|---|---|---|
| median AUC, ID / OOD | 0.872 / 0.844 | — |
| AUC drop ID − OOD: mean, median, q25 | **0.020**, 0.016, −0.002 | 0.033, 0.036, 0.007 |
| models with OOD AUC above ID AUC | 28.7% | 21.3% |
| training accuracy on own subsample: median; models ≥ 0.99 | 0.851; 42 (xgb 14, et 12, rf 12, hgb 4); 16 on all training rows | 0.858; 50 (rebuilt from v1 data in a 2026-10-14 test; the v1 cell does not store it) |

The aggregate shift is smaller on v2. O1 should not promise a large drop. The interpolation check
has 258 non-interpolating models to work with.

**C3. Selections** (paper protocol; 24 tasks in 23 min). `aotil/audit_selection.py`: 33 of 33
checks pass. Every θ file is a finite (12,771,) vector; test r recomputes to 1e-16 with the same
code and to about 1e-7 in independent float64; coherence reproduces at all 12 budgets.

Aggregate ID→OOD r over all 12,771 OOD points: **+0.880** [0.81, 0.93] on the 60 test models
(v1 +0.784); +0.829 over all 300 (v1 +0.714). The largest subset with test r ≤ −0.3 is 7,500 of 12,771
(58.7%) for both variants.

| S | classic r [95% CI], purity, components, tiles | spatial μ=3 k=10 | smoothed classic |
|---:|---|---|---|
| 250 | −0.841 [−0.90, −0.75], 0.16, 143, 23 of 24 | −0.739 [−0.84, −0.60], 0.73, 17, 11 | −0.628, 0.69, 23, 11 |
| 500 | **−0.889** [−0.93, −0.82], 0.21, 221, 23 | **−0.763** [−0.85, −0.63], 0.81, 20, 12 | −0.553, 0.69, 43, 18 |
| 1,000 | −0.886 [−0.93, −0.82], 0.31, 291, 24 | −0.794 [−0.87, −0.68], 0.85, 31, 13 | −0.614, 0.70, 70, 21 |

Coherence costs 0.09–0.13 in r at S = 250–1,000 (v1: a gain at S = 250). Smoothing the classic
weights gives worse r and lower purity with more components, so the penalty, not smoothing in
general, buys the coherence.

**C4. Regions** (`aotil/regions.py`). Tested on v1 first: it reproduces REGION-ANALYSIS-FLORIDA
§2.1 exactly (S = 500: 8 components, 5 tiles, 198/142/102/27). v2:
- **Spatial S = 250:** 17 components, 11 tiles. Components of at least 20 points:
  - 70 points (26 positive), Hamilton
  - 36 (0 positive), Palm Beach
  - 27 (27 positive), Miami-Dade
  - 26 (25 positive), Okeechobee
- **Spatial S = 500:** 20 components, 12 tiles. Components of at least 20 points:
  - 93 points (42 positive), Hamilton
  - 57 (49), Miami-Dade
  - 51 (10), Palm Beach
  - 42 (6), Dixie
  - 37 (20), Okeechobee
  - 37 (25), Indian River
  - 32 (16), Marion
  - 20 (18), Nassau–Duval

  All are low-HAND (median 0.4–2.4 m) wetland or rural land.
- **Classic S = 500:** 221 components across 23 tiles; one component reaches 20 points.
- Some regions are mostly band negatives: S = 250 Palm Beach has 0 of 36 positive, S = 500
  Dixie 6 of 42. The region tables now carry `cell`, `variant`, `S` and `seed` columns.
- **Recurring from v1:** Miami-Dade and Okeechobee / Indian River. The North Florida region moves
  from Columbia and Bradford–Union to neighbouring Hamilton.

Still to run (commands in `docs/RUN-FL-V2.md`): seeds 1–2; the μ sweep. Null-select and the
family-disjoint split need new flags in `aotil/oodselect_paper.py`.

## 7. Part D: GIS inputs for the boundary analysis

| item | result | verdict |
|---|---|---|
| D0 `check_gis_inputs.py` | 26 ok. `haz_FL_5070`: 592,506 polygons, sampled 30,000/30,000 valid, EPSG:5070 confirmed from idx bounds; every region positive's polygon found and containing it; `pan_FL` 7,368 panels (1 repaired). Missing: only NWI, PLSS, roads, non-NHDPlus canals (optional D6) | ✓ |
| D1 `data/interim/analysis/point_extras_FL.parquet` | 80,000 rows × 10: `point_id`, `meta_firm_pan`, `meta_firm_pan_n`, `meta_firm_pan_eff_date`, `meta_near_src_row_id`, `meta_near_fld_ar_id`, `meta_near_fld_zone`, `meta_near_dist_m`, `meta_edge_src_row_id`, `meta_edge_dist_m` (edge columns on positives only). All 40,000 positives covered by their polygon; nearest-SFHA distance equals the dataset's exactly; panel dates agree for 79,945/79,945. `meta_edge_dist_m` median 175 m (A), 281 m (AE) | ✓ |
| D2 `aotil/lobes.py` | S = 250: 4 components, 8,792 lobes (809 region, 2,200 matched_A, 5,783 AE), 740,756 vertices. S = 500: 8 components, 20,939 lobes (1,104 region, 2,601 matched_A, 17,234 AE), 1,885,012 vertices (331,048 region, 521,098 matched_A, 1,032,866 AE). Unit inward normals pass the per-component assertion. Contact flags mark shared SFHA edges. Analysis bboxes (all lobe extents + 3 km): S = 500 sum 63,897 km² (union 58,399; region-lobe-only boxes 15,846). Matched unselected Zone A lobes fall short of 3 per region lobe in 7 of 12 components (e.g. S = 500 c1: 1 of 9) | ⚠ decision 4 |
| D3 `pull_dem.py` | 3DEP 1/3″ via `--remote` COG reads: at most 2.68 GB read; 8 rasters built and 4 shared as hardlinks, 1.3 GB on disk, 8 min. EPSG:5070, 10 m. Three boxes are 76–90% valid (ocean), one 98.8%, the rest 100% (including the Miami-Dade box, where 3DEP carries values over Florida Bay). All 740,756 and 1,885,012 vertices have `z`, `z_in`, `z_out`. 6 tiles were republished in 2025–26; Earth Engine's version of them is not checked | ✓ |
| D4 `pull_ssurgo.py` | Soil Data Access: 197,241 unique map-unit polygons, 800 MB in 207 requests, 8 min; 1.1 GB on disk. S = 500: 501,292 soil-line segments (297,246 km), 4,673 survey-area joins, 32,102 water edges; perimeter balance 1.0000–1.0001. `muaggatt` attributes on every polygon and the differing ones on every line. But the log warns that polygons are not topologically clean in 4 of the 8 S = 500 components (c0, c3, c6, c7; c6 has 30 unresolved probes and 90 in overlapping polygons) and in S = 250 c0; 14 vertices fall in no map unit (`soil_ok` False). Inspect those components before using class S | ⚠ |
| D5 `pull_gis_lines.py` | TIGER/Line 2026 counties (83 MB): 146 county lines and 66 state or coast lines, perimeter gap 0 m. FIRM panel edges: 16,244 lines, gap −0.14 m | ✓ |
| D6 optional | canals from NHDPlus CanalDitch (63,595 flowlines in Florida's footprint); NWI, PLSS, roads not built | ✗ optional |
| D7 `aotil/edge_layers.py` | `results/2026-10-14/boundary/*_vertex_layers.parquet` (56 columns): distance to and id of the nearest soil line (class S), shore, county survey join, other survey-area join, county line, state or coast line, own-DFIRM panel edge (interior and exterior), and canal; plus which soil attributes change. Two flags, added in this run after the first attempt refused to measure: `outside_state` (375 St. Marys vertices up to 215 m past the TIGER line) and `outside_own_panels` (628 vertices of one Hendry Zone AE lobe that reaches into Lake Okeechobee). No analysis | ✓ |

## 8. Experiment → data → status (E1)

| experiment | data | status | path |
|---|---|---|---|
| O1 shift | per-model ID/OOD AUC, seed 0 | ✓ built (mean drop 0.020); seeds 1–2 and a tile bootstrap to do | `data/interim/aotil/cells/FL_band_AE_s0_models.csv` |
| O2 aggregate AotL | `r_full_test` | ✓ +0.880 | `results/2026-10-14/oodselect_paper/` |
| O3 classic OODSelect + coherence | classic θ, 12 budgets | ✓ | `…/classic/` |
| O4 Laplacian vs classic | spatial θ; μ sweep | ✓ μ = 3; μ sweep to run | `…/spatial_mu3_knn10/` |
| smoothed-classic ablation | post hoc | ✓ | `…/FL_band_AE_s0_selection_audit.md` |
| regions | components with `point_id` | ✓ seed 0 | `results/2026-10-14/regions/` |
| O5 boundary analysis | lobes, 30 m vertices with z, soil/county/panel/canal distances | ✓ data built; the analysis is next | `results/2026-10-14/boundary/`, `data/interim/gis/` |
| null-select, family-disjoint split | — | ✗ flags missing in `oodselect_paper.py` | — |
| interpolation check | training acc/AUC per model | ✓ | `_models.csv` |
| Contribution 3 map | regions from all seeds + classified edges | ✗ needs seeds 1–2 and the analysis | — |
| documentary check | FIS reports for the v2 region counties | manual (Victor) | — |

## 9. Numbers in the paper that the rebuild changes

| sentence in `acl_latex.tex` (Introduction) | v1 | v2 |
|---|---|---|
| "We assemble **98,578 points** across Florida with **45 covariates** from **seven public sources**." | 98,578 | **80,000**; 45 and seven unchanged |
| "We train **300 models** on Zone AE points and evaluate them on Zone A points." | 300 | 300 |
| "the 300 models come from **eight families**" | 8 | 8 |

For Methods and Results: band subset 62,578 → 80,000; split 22,658 / 7,634 / 9,412 → 28,109 /
10,235 / 12,771 (71 / 24 tiles); positive share ID / OOD 0.534 / 0.639 → 0.403 / 0.476; mean AUC
drop 0.033 → 0.020; aggregate r +0.784 → +0.880; classic S = 500 r −0.718, purity 0.26,
186 components, 22 tiles → −0.889, 0.21, 221, 23; spatial S = 500 r −0.635, 0.91, 8, 5 →
−0.763, 0.81, 20, 12. Methods also needs the state rule for band negatives (§5 text), GSW v1.4,
and Atlas 14 Volume 9.

## 10. Scripts and runtimes on this laptop

| script | runtime this run |
|---|---|
| `archive_sampling_v1.sh` | seconds |
| `pull_gis_lines.py` (TIGER + panels) | 9 s incl. download |
| `sample_points.py FL …` / `--audit` | 23 min / 1 s |
| `pull_gee.py FL` / `--merge FL` | 2.4 min / seconds |
| `pull_atlas14.py FL` / `--merge FL` | 2 min (all 35 requests failed: 503) / seconds |
| `prep_nhdplus_units.py`, `pull_hydro.py` stages and merge | seconds each |
| `audit_dataset.py` | 35 s |
| `aotil/zoo.py` (300 models) / `audit_cell.py` | 22 s / 7 s |
| `run_oodselect_local.sh` (24 tasks, 8 jobs) / `audit_selection.py` | 23 min / 1 s |
| `aotil/regions.py` | 2 s |
| `pull_point_extras.py` | 140 s |
| `check_gis_inputs.py` | 12 s |
| `aotil/lobes.py` | under 1 min per budget |
| `pull_dem.py fetch --remote` / `sample` | 8 min / 4–8 s |
| `pull_ssurgo.py` plan / fetch | 3 min / 8 min |
| `aotil/edge_layers.py` | 92 s (S = 250), 257 s (S = 500); peak 4.3 GB (from the first, refused run) |

## 11. What stays manual

- FEMA Flood Insurance Study reports for the v2 region counties: Hamilton, Miami-Dade, Palm
  Beach, Dixie, Okeechobee, Indian River, Marion, Nassau, Duval.
- USGS OFR 74-1111, if a recurring region overlaps the Osceola National Forest area (Hamilton's
  region is next to it).
- PDFs for `collins2022` and `jia2026`, not in `refs/cited/`.

## 12. Open items and decisions for Victor

1. **Atlas 14 gap (fix when PFDS is back).** Steps:
   - run `pull_atlas14.py FL`, then `--merge FL`, then `pull_hydro.py --merge FL` (restores
     the column order). Blocks over open water may still fail as outside Atlas 14's area.
   - rerun from step 7 into a new results date. Add `--overwrite` to `audit_dataset.py`,
     `zoo.py` (the cell path is not dated), `pull_point_extras.py` and `check_gis_inputs.py`,
     or move the old outputs aside first.

   Until then, 191 points (0.24%, mostly AE band negatives) carry NaN rainfall features.
   Recommended: fill it before seeds 1–2, so all three seeds use one dataset.
2. **Open-water band negatives** make missing values a label cue (§5). Recommended: keep the
   protocol; report missingness by stratum, and AUC on non-water points as a sensitivity.
3. **`channel_slope` −9998** (43 points). Keep (v1 parity, disclose), or map it to NaN in
   `pull_hydro.py --merge` (a one-line change, not made).
4. **Boundary-analysis design choices.** The data for both options is already collected:
   - the bbox rule covers all lobe roles (3.0× the region-only area at S = 250, 4.0× at S = 500);
   - matched lobes fall short of 3 per region lobe in 7 of 12 components;
   - small "near-rule" polygons count as region edge (`from_positive` separates them);
   - perimeter matches for large region lobes are poor (`match_dlogp` stored);
   - many ±100 m probes leave their lobe (probe flags stored);
   - class S = `soil_line` only, with survey-area joins separate.
5. **Aggregate r in the paper:** `r_full_test` on the 60 test models (+0.880) is recommended.
   Name it in Methods.
6. **Null-select and the family-disjoint split** need flags in `aotil/oodselect_paper.py`.
7. **Smaller items.**
   - lon/lat are NAD83.
   - xgb table means are float32 (1e-7).
   - Prose for Methods (`docs/FEATURE-SOURCES.md` §7): NLCD 2021 is Dewitz 2023; Florida
     the catchment holding a typical point has a median area of 24.7 km² (per distinct
     catchment 3.6 km²), so StreamCat values are shared over far more than "1–3 km²"; reach
     attributes come from the nearest reach.

## 13. Checksums and scope (E3)

Before the run, Victor limited the project to Florida, deleted the other nine states' files
(`data/interim/prep`, `points`, `gee`, `hydro`, `nfhl_states`), and staged the deletion of the
nine tracked `dataset_<ST>.parquet` with `git rm`. The brief's "other nine checksums unchanged"
check therefore no longer applies. What remains: `FEATURES.txt` is byte-identical before and
after (sha256 `31b8654f…8301`), and the non-Florida rows of the shared Atlas 14 cache are
unchanged.
