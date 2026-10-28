*Historical document, kept as written; script paths refer to the camera-ready `causal-graphcast` layout (`src/` and `flagship_sae/` are now `graphcast_sae/`, regrouped by experiment; `notes/` is now `docs/notes/` and `docs/prereg/`; `scripts/` launchers are in `scripts/` here).*

# Amendment 3 to prereg_cluster_revision.md: gate 2 failed under default XLA; switch to deterministic ops

Written 2026-10-12 after the gate run, **before any J1 storm other than Ida and before any J1/J2/J7 bar
was scored.** The only numbers seen at this point are listed below.

## What happened (instrument failure, recorded)

- J1 Ida, default XLA flags (job 23047447, H100): **gate 2 failed.** `gain-1` (an exact no-op patch)
  differed from `baseline`: Δ deepening 0.11 hPa, Δ err 0.13 hPa; `rand-gain-1` differed by 0.10 hPa.
  Gate 3 passed: Ida `gain-0` d_deepen 2.95 vs 2.68 committed (Δ 0.28 < 0.37).
  Kept as `results/skill/suff_ctl_nondet/run_ida2021.npy`, not used for any bar.
- J4 (job 23047819, default flags, as pre-registered): 5 identical copies of an arm in one invocation
  are not bit-identical. Worst single-arm deepening range 0.125 hPa (Patricia baseline), worst
  pairwise trajectory 0.20 hPa. Floor for a difference of two arms: **0.25 hPa**
  (`results/skill/determinism_j4/rev_verdict.json`).
- Diagnostic (job 23047818): same Ida arms with `XLA_FLAGS=--xla_gpu_deterministic_ops=true`.
  `gain-1` == `baseline` and `baseline#r1` == `baseline` **bit-for-bit** (max |Δ| = 0.0). The patch
  is an exact no-op; the default-mode failure is GPU scatter nondeterminism, not a code defect.

## Change

All remaining J1, J2, J6, J7 runs use `--xla_gpu_deterministic_ops=true` (~5x slower per rollout).
Gate 2 is then checked bit-for-bit as originally written. The bars keep the **J4 floor of 0.25 hPa**
as their threshold (conservative: within-run noise is zero under deterministic ops, but the J4 floor
is what a non-deterministic reproduction of these numbers would see).

Patricia: its J4 restore-to-normal effect (~0.2 hPa) is below the 0.25 hPa floor, consistent with
its pre-declared under-resolved status.
