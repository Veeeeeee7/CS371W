Victor has rewritten the Experiments section of the paper, and the local `CS371_Cho_Li/acl_latex.tex` now matches his Overleaf version. The text is the specification: the code must do exactly what it says. Please audit the code against it, change the code where they differ, rebuild the dataset, rerun the full suite, draw the new data figure, and update the paper's numbers. Victor will then rewrite the prose with Claude from your report, so the report has to be complete.

Read `docs/TODO-CLAUDE-CODE.md` first for the standing rules. The full brief is `docs/TASK-REVISED-EXPERIMENTS-2026-11-07.md`. Its §0 lists ten places (D1 to D10) where the text and the code already differ and says what to do with each. Do its parts in order:

- **A.** Audit every statement of the Approach, Experiments and Appendix A against the code (a table with `file:line`). No new results.
- **B.** Code changes D1 to D7: candidate sampling with no cut, no polygon cap, terrain variability within a 300 m radius, binary fluvial class, one-hot encoding, dropping duplicate models, matching on the six numeric features. Tests for each.
- **C.** Archive the point-keyed interim files, rebuild the dataset into `data/processed_v4/`, audit it, and measure the spatial autocorrelation range of every feature. This is a stop point.
- **D.** Fit the new cell and check it. This is a stop point.
- **E.** The full run, as Part C of the 2026-11-06 brief: all 81 searches and everything downstream, claims C1 to C7.
- **F.** The data figure (`figures/fig_feature_boxes.pdf`) from Victor's approved sample, inserted at `% INSERT FIGURE HERE`.
- **G.** Fill the placeholders and update every number in the paper. Numbers only; flag sentences, do not rewrite them.
- **H.** The report, with everything the Data statistics prose needs, and the docs.

**Stops** (see the brief):
- Part A: the text describes a method the code does differently and D1 to D10 do not cover it.
- Part C: a zone-and-label group keeps fewer than 20,000 points; the dataset audit fails; the soil lookup fails its limits.
- Part D: the population checks of the 2026-11-06 brief.
- A grid keyword or value has to change.

**Rules:**
- `export D=<today>`. Outputs under `results/$D/`, logs under `logs/$D/`, data in `data/processed_v4/`, the cell in `data/interim/aotil/cells_v4_$D`.
- The new points reuse point ids and the pull scripts skip existing outputs, so archive the point-keyed interim files first (move, never delete).
- Do not change earlier results, cells or datasets.
- In the paper, change only numbers, placeholders, figure files and the two insertions of Part F.
- Use `.venv/bin/python`. Every new or changed function gets a test.
- Delete retired code; do not comment it out.
- Keep the Mac plugged in with the lid open.
- Do not commit or push.
- Victor comes from CS, not GIS. Explain GIS terms in CS terms in the report.
