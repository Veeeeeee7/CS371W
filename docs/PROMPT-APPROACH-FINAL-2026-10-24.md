Victor and Claude have read and revised the Approach section of the paper. Approach 3.1–3.4 is now final. The final version is saved at `CS371_Cho_Li/approach-3.1-3.4-2026-10-24.tex`, and it matches Overleaf, which is the source of truth. Please bring the codebase and the local paper in line with it.

Read `docs/TODO-CLAUDE-CODE.md` first. It has the standing rules for editing the paper and the two open items. Do them in this order.

**First, item 2: sync the local paper (text only, no new results).**
- In `CS371_Cho_Li/acl_latex.tex`, replace everything from `\section{Approach}` up to `\subsection{Region Analysis}` with the final file, verbatim. Do not reword it.
- Leave 3.5 Region Analysis as it is. Victor has not reviewed it yet.
- Switch the preamble from `algpseudocode` to `algorithmic`, as the doc says.
- Remove every section or subsection `\label` and every `Section~\ref` in the file. Rewrite those sentences so they read without the pointer. Do not touch equation, figure, table, or algorithm labels.
- Check `custom.bib`, compile, and confirm there are no undefined references.

**Then, item 1: switch to the squared budget penalty and rerun.**
- The final Approach states the squared penalty $B_S(\mathbf{s})=(S-\|\mathbf{s}\|_1)^2$ from the published OODSelect paper (Eq. 5), not the released code's softplus. Change the code to match, behind the `--budget-penalty` flag described in the doc. Keep everything else in the search unchanged.
- Use option b for the λ range: Eq. 5 as written, with λ searched over a wider log-uniform range. Do not use option c without asking Victor.
- Before the full reruns, do the quick check at S = 500 for classic and spatial. If the soft total ends near S and the test correlation still inverts, go on. If the penalty swamps the correlation or the inversion disappears, **stop** and report what you saw.
- Rerun everything the doc lists, on the same v3 cells and seed-0 split.
- Write `docs/RUN-SQUARED-PENALTY-<date>.md` with every paper number, old and new, and whether C3 to C6 still hold. **Stop** before changing any number in the paper. Victor decides.

**Rules.**
- `export D=<today>`. Outputs go under `results/$D/` and logs under `logs/$D/`. Do not change earlier results or cell folders.
- Use `.venv/bin/python`. Every new or changed function gets a test.
- Do not commit or push.
- When an item is done, move it to Done in `docs/TODO-CLAUDE-CODE.md`, with the date and the report it produced.
- Victor comes from CS, not GIS. Explain GIS terms in CS terms in anything you write for him.
