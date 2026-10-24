*Historical document, kept as written; script paths refer to the camera-ready `causal-graphcast` layout (`src/` and `flagship_sae/` are now `graphcast_sae/`, regrouped by experiment; `notes/` is now `docs/notes/` and `docs/prereg/`; `scripts/` launchers are in `scripts/` here).*

# Amendment 2 to prereg_cluster_revision.md: J2 (direction baselines)

Frozen 2026-10-12, before any J2 number exists.

**Question.** Does the SAE convection direction matter, or only the size of the push?

**Patch.** `SAEJax.delta_dir`: per node, the treatment's delta_gain vector
Δ = Σ_f (g−1)·max(f−f_normal, 0)·d_f has norm ‖Δ‖; the baseline arm applies ‖Δ‖·u along a fixed
unit direction u instead. Per-node norm, mask, schedule and gain are identical to the treatment by
construction; only the direction differs.

**Arms**, one invocation per storm, 8 NH storms, g ∈ {1.5, 2, 3}:
- `gain-g`: the SAE decoder directions (reference, = J1's treatment).
- `dir-rand{k}-g`, k = 1..5: random unit directions, fixed seeds 0..4, orthogonalised against the
  span of the three convection decoder columns (B1).
- `dir-dom-g`: difference of means (B3), u ∝ mean over disk nodes of [layer-8 activation at the IC
  − mean activation at the same nodes on the storm-free analogue dates that the run already uses
  for f_normal], normalised. **Deviation from spec:** computed in-run from those analogues, not
  from the i.i.d. dump (unavailable on this cluster).

**Readout and reporting.** Deepening-error reduction vs ERA5 (err_red, as J1). Per storm:
treatment − median(B1) and treatment − B3 at each gain, plus each arm at its own best gain (the same
in-sample selection for every arm, labelled in-sample). Report the median over the 7 developing
storms with a bootstrap 95% CI over storms (10,000 resamples, seed 0), whichever way it comes out.

- "Direction-specific" iff treatment − median(B1) at g = 2 has a CI excluding 0 (positive).
- B3 ≥ treatment is reported as a result (AxBench-style), not as a failure.
- If every arm, including the random ones, gives similar err_red, the paper says the lever is the
  size of the push in the storm core, not the SAE direction.
