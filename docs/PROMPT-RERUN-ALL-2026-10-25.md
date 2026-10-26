Victor has finished the paper's Approach section (3.1 to 3.6 and Algorithm 1 in the local `CS371_Cho_Li/acl_latex.tex`). It is now final. Please rerun every result in the paper so that it matches the Approach, add the new region analysis and the GTV loss, and update the paper.

Read `docs/TODO-CLAUDE-CODE.md` first for the standing rules. The full brief is `docs/TASK-RERUN-ALL-2026-10-25.md`. Do its parts in order:

- **A.** Audit the code against the Approach.
- **B.** Add GTV (`--reg gtv`).
- **C.** Write `aotil/region_match.py`, the caliper matching of Approach 3.6, and check it against Claude's sandbox numbers.
- **D.** Rerun everything from the model cell on.
- **E.** Update the paper.
- **F.** Write the report and update the docs.

When the brief and the code disagree with the Approach, the Approach wins. When anything is unclear, read the paper's Approach section. Do not edit the Approach.

**Stops** (see the brief):
- The refit cell does not reproduce `cells_v3`.
- `region_match.py --exclude own-tile` misses the sandbox numbers by more than 0.01.
- C6a or C6b fails. Stop before the Abstract, Introduction and Error Analysis edits.
- An audit fix would change a method choice rather than fix a bug.

**Rules:**
- `export D=<today>`. Outputs go under `results/$D/` and logs under `logs/$D/`. The refit cell goes in `data/interim/aotil/cells_rerun_$D`.
- Do not change earlier results.
- Use `.venv/bin/python`. Every new or changed function gets a test.
- Keep the Mac plugged in with the lid open.
- Do not commit or push.
- Victor comes from CS, not GIS. Explain GIS terms in CS terms in the report.
