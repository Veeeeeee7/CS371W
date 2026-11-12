Victor has changed how the 300 models are built. Models now differ only by family and hyperparameter setting: every model trains on all features and all training points, and each family has a fixed grid with some deliberately weak settings. Models that predict one class for almost every point are dropped; models that happen to share predictions are kept. Please rebuild the model cell this way, do a full run of the subset search on the new models, rerun every result in the paper on it, and update the paper's numbers.

Read `docs/TODO-CLAUDE-CODE.md` first for the standing rules. The full brief is `docs/TASK-ZOO-GRIDS-2026-11-06.md`. Do its parts in order:

- **A.** Change `aotil/zoo.py` to the fixed grids and the degenerate drop rule (duplicates stay), and update every reader that assumes 300 models or round-robin families.
- **B.** Fit the new cell and check it. This is a stop point.
- **C.** A full run of the search on the new models: every budget in the grid for the classic, Laplacian and GTV searches, with the same settings and seeds as 2026-10-25, then the mu sweep, all controls, the regions, the matching and the figures, as in Part D of the 2026-10-25 brief. Check the claims.
- **D.** Update the paper's numbers and fill the `\textbf{XX}` placeholders. Numbers only, no prose.
- **E.** Write the report and update the docs.

Victor is editing the paper in Overleaf while you work. In the local `CS371_Cho_Li/acl_latex.tex`, change only numbers, placeholders and figure files. Put every change in the report as a Before/After block he can paste, and flag any sentence whose claim stops holding instead of rewriting it.

**Stops** (see the brief):
- Part B: kept N < 200, lowest kept ID accuracy above 0.60, a family below 10 models, or probit r(ID, OOD) below 0.5.
- A grid keyword or value has to change.
- The mu rule no longer picks mu = 3.
- C6a or C6b fails. Stop before any paper edit.

**Rules:**
- `export D=<today>`. Outputs go under `results/$D/`, logs under `logs/$D/`, and the cell in `data/interim/aotil/cells_grid_$D`.
- Do not change earlier results.
- Use `.venv/bin/python`. Every new or changed function gets a test.
- Delete retired code; do not comment it out.
- Keep the Mac plugged in with the lid open.
- Do not commit or push.
- Victor comes from CS, not GIS. Explain GIS terms in CS terms in the report.
