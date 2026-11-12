Victor wants to know what happens to every result when the models also see the two soil predictors of Woznicki et al. (2019): the SSURGO flood-frequency class and the fluvial-soil class. They were left out because the soil maps were the held-out layer of the boundary analysis, which is no longer in the paper. Zone A outlines were drawn partly from soil maps, so these features share a source with one domain's labels, and the result could go either way. Please pull the two features for every point, build a 9-feature dataset, fit the grid population on it, rerun the full suite exactly as on 2026-11-06, and compare every result with the 7-feature run.

This is an experiment, not a switch. The 7-feature results stay the paper's results. **Do not edit anything in `CS371_Cho_Li/`.**

Read `docs/TODO-CLAUDE-CODE.md` first for the standing rules. The full brief is `docs/TASK-SOIL-W9-2026-11-07.md`. Do its parts in order:

- **A.** Pull the soil class of every point from NRCS Soil Data Access (reuse the `SDA` class of `pull_ssurgo.py`), cross-check it, build `data/processed_w9/`, and compare soil classes with labels by zone. This is a stop point.
- **B.** Make the code read the feature count from `FEATURES.txt` (at least `aotil/region_match.py`), with defaults that reproduce 2026-11-06 exactly.
- **C.** Fit the w9 cell and check it. This is a stop point.
- **D.** A full run on the w9 cell, as Part C of the 2026-11-06 brief: all 81 searches and everything downstream, claims C1 to C7.
- **E.** Compare w7 and w9: every paper number side by side, what happened to the eight 2026-11-06 regions, selection overlap, the new regions, feature importance, and the verdict by the rules fixed in the brief.
- **F.** Write the report and update the docs.

**Stops** (see the brief):
- Part A: more than 2% of points in no map unit, an unexpected flood class, more than 5% of non-water points with no flood class, the cross-check below 99%, or a point lookup that would take more than about 8 hours.
- Part C: the stops of the 2026-11-06 brief's Part B.
- A grid keyword or value has to change.

**Rules:**
- `export D=<today>`. Outputs go under `results/$D/w9/`, logs under `logs/$D/w9/`, the cell in `data/interim/aotil/cells_w9_$D`, the features in `data/processed_w9/`.
- Do not change earlier results, cells, `data/processed/` or `data/processed_w7/`.
- SDA is a shared government server: at most 2 concurrent requests, with retries and a cache.
- Use `.venv/bin/python`. Every new or changed function gets a test.
- Delete retired code; do not comment it out.
- Keep the Mac plugged in with the lid open.
- Do not commit or push.
- Victor comes from CS, not GIS. Explain GIS terms in CS terms in the report.
