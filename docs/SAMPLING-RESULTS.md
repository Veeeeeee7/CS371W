# Sampling results, and the design change they force

`sample_points.py` run on all ten states, 2026-09-19.
**878,336 points, 16,397 blocks, every stratum exactly balanced between domains.**

| state | total | blocks | DFIRMs | pos/domain | band neg/domain | shadedX neg/domain |
|---|---:|---:|---:|---:|---:|---:|
| AZ | 116,630 | 1,840 | 15 | 19,806 | 19,457 | 19,052 |
| TX | 116,382 | 3,678 | 134 | 19,955 | 19,848 | 18,388 |
| IA | 101,028 | 1,613 | 99 | 19,837 | 19,928 | 10,749 |
| LA | 100,246 | 1,153 | 52 | 19,936 | 13,900 | 16,287 |
| MO | 98,904 | 1,767 | 98 | 19,915 | 19,949 | 9,588 |
| FL | 98,578 | 1,709 | 67 | 19,963 | 11,326 | 18,000 |
| MN | 96,628 | 1,644 | 64 | 19,824 | 19,882 | 8,608 |
| CO | 65,592 | 1,411 | 45 | 14,162 | 14,508 | 4,126 |
| NC | 57,300 | 1,381 | 99 | 10,082 | 10,026 | 8,542 |
| VT | 27,048 | 201 | 13 | 6,093 | 6,357 | 1,074 |

CO, NC and VT are supply-limited below the 20,000 target — the states simply do
not contain 20,000 points 200 m apart inside one zone class.

---

## 1. The finding: balanced counts, unbalanced geography

Point counts match exactly by construction. **Spatial coverage does not.**

| state | blocks with A | blocks with AE | ratio | blocks with BOTH | % of blocks |
|---|---:|---:|---:|---:|---:|
| IA | 1,326 | 352 | **3.77** | 229 | 15.8 |
| CO | 1,005 | 314 | 3.20 | 243 | 22.6 |
| AZ | 1,191 | 392 | 3.04 | 242 | 18.0 |
| MO | 1,386 | 491 | 2.82 | 274 | 17.1 |
| TX | 2,376 | 930 | 2.55 | 334 | 11.2 |
| MN | 1,053 | 477 | 2.21 | 239 | 18.5 |
| LA | 740 | 559 | 1.32 | 285 | 28.1 |
| VT | 158 | 136 | 1.16 | 120 | **69.0** |
| FL | 918 | 978 | 0.94 | 446 | 30.8 |
| NC | 138 | 978 | **0.14** | 85 | 8.2 |

Splits are made at block level, so **the block count is the effective sample
size**, not the point count. Iowa's AE side has 352 independent units against the
A side's 1,326. The two domains are not the same size in the way that matters.

And in eight of ten states, **fewer than a third of blocks contain both domains.**
Zone A is dispersed; Zone AE is concentrated along studied river corridors and
coasts. Detailed studies get done where people and money are.

### Why this is not just a nuisance

A naive A→AE transfer measures two things at once:

1. **provenance shift** — how the label was made (the thing we care about)
2. **geographic shift** — where the label exists at all (a confound)

They are entangled by construction, because FEMA decides *where* to do a detailed
study using the same considerations that decide *how well*.

---

## 2. The design change: paired within-block as the primary contrast

**Primary analysis.** Restrict to blocks containing both domains. Geography is held
fixed; the only thing varying is how the map was made. This is the clean experiment
and it is what the headline transfer gap should be measured on.

**Secondary analysis.** The full-state contrast, which carries both shifts.

**The difference between them is a result, not an error bar.** It decomposes the
gap into *where the studies are* versus *how the studies were done* — which is
exactly the explanation the OODSelect paper could not produce for its own subsets.
DISDE (arXiv:2303.02011) formalises this decomposition and requires shared support,
which the paired blocks provide and the full-state contrast does not.

**Testbeds.** Vermont at 69% shared blocks is the cleanest, though small (120
paired blocks). Florida at 31% and 446 paired blocks is the best large one. Iowa at
15.8% is where the two analyses should diverge most, and it is also the state with
a qualitatively different Zone A (Base Level Engineering risk classes), so it is
the sharpest test of whether the gap tracks provenance or geography.

**NC stays out of the primary analysis.** 138 A blocks against 978 AE, 8.2% shared,
and only 334 Zone A source polygons contributing. It remains useful as a
degenerate-case check.

---

## 3. Pseudo-replication, stated plainly

| state | A: polys → pts/poly | AE: polys → pts/poly |
|---|---|---|
| NC | 334 → 30.2 | 994 → 10.1 |
| LA | 374 → 53.3 | 605 → 33.0 |
| AZ | 580 → 34.1 | 900 → 22.0 |
| FL | 597 → 33.4 | 1,005 → 19.9 |
| MO | 612 → 32.5 | 513 → 38.8 |
| VT | 722 → 8.4 | 405 → 15.0 |
| TX | 791 → 25.2 | 1,051 → 19.0 |
| MN | 822 → 24.1 | 577 → 34.4 |
| IA | 838 → 23.7 | 591 → 33.6 |
| CO | 1,252 → 11.3 | 849 → 16.7 |

Twenty thousand points per domain come from a few hundred to ~1,250 source
polygons. The largest 1% of contributing polygons supply 8–26% of a domain's
points. Any confidence interval computed as though the points were independent
will be far too narrow; **bootstrap at the block level**.

---

## 4. Provenance gradients that survived sampling

Median values on positives, per state:

| state | map scale A | map scale AE | BFE km/km² A | BFE km/km² AE | eff. year A | eff. year AE |
|---|---|---|---:|---:|---:|---:|
| VT | 1:12,000 | 1:6,000 | 2.098 | 2.098 | 2008 | 2008 |
| MN | 1:24,000 | 1:12,000 | 0.080 | 0.682 | 2017 | 2015 |
| MO | 1:24,000 | 1:12,000 | 0.137 | 0.572 | 2011 | 2012 |
| AZ | 1:24,000 | 1:12,000 | 0.446 | 1.133 | 2009 | 2013 |
| CO | 1:24,000 | 1:12,000 | 0.699 | 0.699 | 2016 | 2019 |
| IA | 1:24,000 | 1:12,000 | 0.051 | 0.182 | 2019 | 2021 |
| TX | 1:24,000 | 1:24,000 | 0.090 | 0.471 | 2011 | 2016 |
| LA | 1:24,000 | 1:24,000 | 0.116 | 0.058 | 2012 | 2012 |
| FL | 1:12,000 | 1:12,000 | 0.143 | 0.174 | 2013 | 2014 |
| NC | 1:12,000 | 1:12,000 | 0.302 | 1.074 | 2006 | 2015 |

Two things worth noting.

**Map scale separates the domains in six of ten states**, typically 1:24,000 for A
against 1:12,000 for AE. In Vermont, where `STUDY_TYP` is 100% unpopulated and
carries nothing, scale still splits them 1:12,000 / 1:6,000. It is a clean
continuous provenance measure — and therefore a clean leak. It stays `meta_`.

**BFE line density is the better gradient.** It varies by 40× across states
(0.051 in Iowa to 2.098 in Vermont) and by up to 8.5× between domains within a
state (Minnesota). Unlike the binary A/AE split it is continuous, so the transfer
gap can be regressed on it rather than compared across two buckets — which is the
form the "explain the subsets" claim needs.

**Louisiana inverts.** Its Zone A has *higher* BFE density than its Zone AE (0.116
vs 0.058). Worth checking before it turns up unexplained in a result.

---

## 5. What this changes downstream

1. Every train/test split, CV fold and OODSelect subset is made on `block_id`.
2. The headline A→AE gap is computed on **paired blocks only**.
3. The full-state gap is reported alongside it, and the difference is interpreted
   as the geographic component.
4. Confidence intervals come from a **block-level bootstrap**.
5. `lon` and `lat` are join keys, **not features** — with block-level splits a
   model given raw coordinates would memorise geography directly. Exclude them
   alongside `meta_*`.
6. NC is excluded from the primary analysis; CO, NC and VT carry a
   supply-limited footnote.
