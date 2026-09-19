# Covariate audit: what the pulled data actually says

Run on a 40,000-point sample (2 shards per state) drawn from the completed
445-chunk Earth Engine pull, 2026-09-19. Every number below is measured, and the
three code changes it forced are already in `pull_gee.py --merge`.

**Headline: the experiment is viable, but only after removing elevation and fixing
the fold design.** The flood signal holds at 0.90 AUC while domain separability
falls to 0.71. That gap is what makes the provenance question answerable; without
these two fixes it was 0.93 against 0.86, which would not have been.

---

## 1. The pull itself is clean

| band | null % | zero % | p1 | p50 | p99 |
|---|---:|---:|---:|---:|---:|
| elev | 0.08 | 1.0 | 0.000 | 283.5 | 2678.5 |
| slope | 0.08 | 5.9 | 0.000 | 0.987 | 34.1 |
| aspect | 0.08 | 6.0 | 0.000 | 164.3 | 355.6 |
| curvature | 0.08 | 5.8 | −7.935 | 0.000 | 9.267 |
| tpi_300 | 0.08 | 3.5 | −36.7 | −0.045 | 33.1 |
| tri_300 | 0.07 | 3.1 | 0.000 | 1.165 | 40.9 |
| hand | 0.98 | 18.4 | 0.000 | 2.400 | 255.6 |
| log1p_upa | 0.98 | 0.0 | 0.006 | 0.013 | 7.700 |
| tpi_broad | 0.98 | 1.6 | −79.7 | −0.208 | 71.8 |
| landcover | 0.12 | 0.0 | 11 | 71 | 95 |
| impervious | 0.12 | 90.2 | 0 | 0 | 75 |
| gsw_occurrence | 0.00 | 90.0 | 0 | 0 | 98 |
| water_frac_900m | 0.00 | 56.3 | 0.000 | 0.000 | 1.000 |

Null rates are 0.08% for 3DEP, 0.98% for MERIT, 0.12% for NLCD — nothing to worry
about. `impervious` and `gsw_occurrence` are ~90% zero, which is what land looks
like; both are legitimately zero-inflated, not degenerate. `hand` sits at exactly
0 for 18.4% of points, meaning "on the drainage network", which is real.

The selftest's own sanity line was partly wrong, and that is worth recording. It
claimed the valley bottom should have the lowest `tpi_300`; the hillslope point
came in lower (−4.497 vs −1.123). That is not a data fault — the "hillslope" test
point has `landcover` 22 and `impervious` 42, so it is a developed, graded site and
a poor terrain reference, and there is no reason three arbitrary points should order
themselves on local relief. `hand` (2.9 / 45.6 / 6.5) and `log1p_upa` did behave as
stated.

---

## 2. Three fixes, each forced by a measurement

### 2.1 Elevation is a location fingerprint

**Absolute elevation alone identifies which of the ten states a point is in with
58.3% accuracy against a 10% baseline.**

On the A-vs-AE domain classifier it contributes **0.264 of 0.85 AUC by permutation
importance — five times the next feature**:

| feature | AUC drop when permuted |
|---|---:|
| elev | 0.2637 |
| water_frac_900m | 0.0756 |
| tpi_broad | 0.0195 |
| tri_300 | 0.0104 |
| log10_upa | 0.0055 |

Median elevation is 294 m for Zone A and 232 m for Zone AE. Pooled across states
that is largely a state indicator.

What dropping it costs and buys:

| | with elev | without | change |
|---|---:|---:|---:|
| positives vs band negatives | 0.932 | 0.919 | **−0.014** |
| positives vs shadedX negatives | 0.899 | 0.796 | −0.103 |
| A vs AE separability | 0.860 | 0.751 | **−0.109** |

Fourteen thousandths of AUC on the primary task to remove a hundred and nine
thousandths of confound. What floods is *relative* elevation, and `hand`,
`tpi_300` and `tpi_broad` already carry it. `elev` is now `meta_elev`. The one
legitimate reason to want absolute elevation — orographic and climate effects — is
better served by StreamCat's precipitation and runoff metrics.

The shadedX column is its own warning: a tenth of an AUC of that protocol's
apparent performance was elevation lookup, because 0.2%-chance polygons sit
immediately above the 1% floodplain and absolute elevation acts as a local
threshold. **Treat `band` as the primary negative protocol and shadedX as the
sensitivity arm**, not the reverse.

### 2.2 Block-level GroupKFold is not spatial cross-validation

Two adjacent 10 km blocks can land in different folds, so a model can memorise a
neighbourhood and then be scored on it. Grouping by progressively coarser tiles:

| task | 10 km blocks | 30 km | 50 km | 100 km |
|---|---:|---:|---:|---:|
| pos vs band, full | 0.932 | 0.924 | 0.911 | 0.898 |
| pos vs band, no elev | 0.919 | 0.913 | 0.909 | **0.904** |
| pos vs shadedX, full | 0.899 | 0.863 | 0.819 | 0.820 |
| pos vs shadedX, no elev | 0.796 | 0.789 | 0.756 | 0.762 |
| **A vs AE, full** | 0.860 | 0.773 | 0.709 | **0.693** |
| A vs AE, no elev | 0.751 | 0.731 | 0.737 | 0.709 |

**Most of what looked like a domain difference was spatial memorisation**: A-vs-AE
falls 0.860 → 0.693. Meanwhile the flood signal without elevation is almost flat
across scales (0.919 → 0.904), which is the signature of a feature set that
generalises rather than memorises.

`tile_30km`, `tile_50km` and `tile_100km` are now columns in the dataset. **Split
on `tile_50km`.** Group counts pooled: 841 blocks → 255 / 167 / 94 tiles, so 50 km
keeps enough groups for 4–5 folds while separating them in space.

### 2.3 Two smaller things

**`log1p_upa` barely transformed anything.** Half the points have an upstream area
under 0.013 km², where log1p(x) ≈ x, so the bulk was crushed against zero with a
few Mississippi-scale outliers holding all the range. `log10_upa` floors at one
MERIT cell (0.0081 km²) and takes a base-10 log: measured on Vermont, the
interquartile spread goes from 0.087 to 1.083, a **12× improvement in usable
range**. No re-pull — the original is recoverable as `expm1`.

**Aspect is undefined on flat ground.** 6% of points returned aspect exactly 0, and
**30.6% sit below 0.5° of slope**, where the value is noise. sin/cos would have
encoded all of that as due north — on floodplains, which is exactly the cells that
carry the positive label. Aspect is now masked to NaN below 0.5°, so roughly a
third of `aspect_sin`/`aspect_cos` are missing by design. Use a model that handles
NaN natively (HistGradientBoosting does) or impute at fit time.

---

## 3. Two confound hypotheses tested and rejected

**"Zone AE is just on bigger rivers."** No. Matching A and AE within six quantile
bins of upstream area leaves separability at **0.853** against 0.850 unmatched, and
dropping `log10_upa` from the feature set entirely leaves it at **0.851**. Stream
size is not the driver. (Median upstream area is 0.018 km² for A and 0.014 km² for
AE — the distributions nearly coincide at the median and differ only in the upper
quartile, 0.435 vs 0.080.)

**"There is no shared support, so reweighting is impossible."** There is. Propensity
overlap on the domain classifier:

| propensity band | share of points |
|---|---:|
| [0.10, 0.90] | 49.4% |
| [0.05, 0.95] | 68.4% |
| [0.02, 0.98] | 82.4% |

68% of points lie in a comfortable overlap region, which is enough for DISDE's
importance weighting on the overlap and honest reporting of what falls outside it.

---

## 4. Where the real difference sits

With elevation gone, the leading discriminator is **`water_frac_900m`** (0.076
permutation importance): median 0.000 for Zone A, 0.087 for Zone AE. Zone AE is
near water that JRC Global Surface Water can see; Zone A is on streams too small
to appear. That is not a nuisance — it is the mechanism, stated in a covariate:
detailed studies get done on mapped rivers. It belongs in the explanation, and it
is also the variable most likely to be doing the provenance/geography double duty,
so report it explicitly rather than letting it hide inside an importance ranking.

Paired-block separability (geography held fixed at 10 km) was 0.789 VT / 0.832 FL /
0.806 IA with elevation included. Those should be recomputed on `tile_50km` folds
without elevation before they go in any write-up.

---

## 5. The honest numbers to quote

Feature set: 13 columns, no absolute elevation, aspect masked on flat ground.
Folds: `tile_50km`, GroupKFold.

| quantity | value |
|---|---:|
| flood susceptibility, positives vs band negatives | **0.904–0.909** |
| flood susceptibility, positives vs shadedX negatives | 0.756–0.762 |
| Zone A vs Zone AE separability | **0.709–0.737** |

The 0.90 figure is comparable to published flood-susceptibility work while being
honest about spatial structure, which most of that work is not. The 0.15-point gap
between the two negative protocols is itself a result: it is the negative-sampling
sensitivity the labeling literature puts at 0.13–0.24 AUC, reproduced here on one
footprint with everything else held fixed.

---

## 6. What to run

1. `python pull_gee.py --merge` — applies all of the above, writes
   `data/processed/dataset_{ST}.parquet` and `data/processed/FEATURES.txt`.
2. Re-run the baselines on the full data with `tile_50km` folds and confirm the
   §5 numbers hold at 20× the sample.
3. Extract `Catchment` and `NHDFlowline` from the NHDPlus archive, subset to the
   ten states, delete the extracted geodatabase.
4. COMID join → StreamCat pull. Priority metrics: `Precip8110Cat` and `RunoffCat`
   (the legitimate replacement for what elevation was proxying), `WetIndexCat`,
   `BFICat`, `ClayCat`/`PermCat`.
5. Distance to flowline, which replaces `water_frac_900m` with a real distance
   rather than a saturating fraction.
