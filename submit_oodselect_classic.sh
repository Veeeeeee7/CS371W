#!/bin/bash
# =============================================================================
# submit_oodselect_classic.sh -- FIRST EXPERIMENT, VERSION 1 OF 2
# OODSelect exactly as the paper ran it: no spatial term (mu = 0).
#
# Salaudeen et al., NeurIPS 2025 (arXiv:2510.24884), reimplemented from the
# authors' released code and checked against it epoch for epoch
# (aotil/test_oodselect_paper.py):
#   * 300-model heterogeneous population per cell, split 60/20/20 into
#     selection / validation / test models (sklearn, random_state=42)
#   * binary correctness Z and ID accuracy, probit on both sides
#   * sigmoid relaxation, softplus budget penalty with cosine warm-up, Adam +
#     CosineAnnealingLR, early stopping on validation r (patience 500),
#     up to 50,000 epochs
#   * 100 Optuna trials per budget over the authors' search space; the best
#     trial by validation r is reported on the held-out test models
#   * the authors' budget grid, 10 ... 7,500 (truncated below the OOD set size)
#   * random (x100) and most-misclassified baselines, Fisher 95% CIs, Spearman,
#     the normalised Jaccard of Eq. 6, and the <1% model-count rule
#
# Cells: FL band, seed 0, both directions (AE->A and A->AE).
# The spatial version is submit_oodselect_spatial.sh; it shares the worker, the
# cells, the model split and the per-budget seeds, and differs in one term.
#
# Run from the repo root on the login node, after ./upload_to_cluster.sh:
#   ./submit_oodselect_classic.sh --date YYYY-MM-DD
#   DRY_RUN=1 ./submit_oodselect_classic.sh --date YYYY-MM-DD    # print only
# Knobs: see oodselect_paper_fanout.sh.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./oodselect_paper_fanout.sh

MU=0
oodsel_submit "$MU" "$@"
