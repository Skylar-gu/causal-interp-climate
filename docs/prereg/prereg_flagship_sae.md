*Copied verbatim from the main repo's `notes/` on 2026-08-25. Script paths refer to the pre-release layout (`flagship_sae/` is now `graphcast_sae/`, regrouped by experiment).*

# Pre-registration — Flagship-SAE side-program: grid-lock detection on 0.25° GraphCast

**Status:** written 2026-08-08, BEFORE any flagship activation is extracted or any
number is looked at. This is the flagship-SAE **side-program** (0.25°/37-level
GraphCast + a pre-trained published SAE). Per the single-model-coherence rule, NOTHING
here is quantitatively cross-comparable to the main G1 causal-graph rung
(graphcast_small, 1°/13-lev). It is a design-sanctioned separate rung in a different
latent space. All outputs tagged `results/flagship_sae_*`.

## What this ports
`sae/retry5_gridlock.py` (small-model grid-lock DETECTION half), re-run on the
flagship layer-8 mesh-node embeddings encoded through the paper's published SAE.
Substrate: merged multi-mesh node **degree** for the flagship mesh (M6, splits=6,
40,962 nodes), reconstructed CPU-only from `graphcast.icosahedral_mesh`.

## Metric (identical form to the small-model test)
For each ALIVE SAE feature f, over a set of teacher-forced flagship windows:
- node signature `featmap[f, n]` = mean activation of feature f at mesh node n.
- content-invariance `CI_f` = between-node SS / total SS (high ⇒ fires at fixed nodes
  regardless of weather content).
- degree-locking `rho_f` = Spearman(featmap[f, ·], node_degree).
- (diagnostic, not a bar) activation-weighted mean degree `wdeg_f`.

## Pre-registered bar (fixed NOW, identical thresholds to the small-model run)
- **Grid-locked feature** := `|rho_f| >= 0.30` AND `CI_f >= 0.50`.
- **PASS** := at least **1** grid-locked feature found among alive features.
- A MISS (zero grid-locked features) is a reportable finding, not a failure to hide.
- Thresholds (0.30, 0.50) are inherited verbatim from `notes/prereg_phase1_2.md` /
  `retry5_gridlock.py` so the flagship result is interpretable on the same scale it
  was designed on. (This is method-level comparability of a detector, NOT a
  quantitative cross-comparison of features/graphs across the two networks — those
  remain forbidden.)

## Secondary, pre-registered as descriptive (no pass/fail)
- Count of grid-locked features and their `(rho, CI, rate, wdeg)` — reported next to
  the small-model outcome (2 candidates found there) as a **qualitative** "does the
  richer M6 multi-mesh, with more degree variation, surface more/clearer grid-lock?"
  narrative only.
- Sensitivity: also report counts at (|rho|>=0.20, CI>=0.40) and (0.40, 0.60) so the
  headline count is not threshold-cherry-picked.

## Sample / windows
- Extract layer-8 flagship embeddings on a small set of teacher-forced ERA5 windows
  (target ~8–24 windows, evenly spaced across seasons, matching the strided regime of
  the small-model dump). Exact N gated on per-forward-pass wall-clock + GPU fit; will
  be recorded in the results JSON. The detector is CPU after extraction.

## Explicitly OUT of scope for this rung (queued, GPU-heavy)
- Causal-inertness ablation (zero each candidate feature during teacher-forced
  flagship steps, confirm no forecast change). Needs many flagship forward passes.
  Queue with a cost estimate; do NOT run here.

## Outputs
- `results/flagship_sae_gridlock.json` (+ `.npy`): candidates, rho/CI arrays, degree
  set, pass/fail, N windows, SAE provenance (which published dict/k/checkpoint).
- `results/flagship_sae_featmap.npy` or S3 if large (dict × 40,962).
- Logs → `out/flagship_sae_*`.
