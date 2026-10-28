*Historical document, kept as written; script paths refer to the camera-ready `causal-graphcast` layout (`src/` and `flagship_sae/` are now `graphcast_sae/`, regrouped by experiment; `notes/` is now `docs/notes/` and `docs/prereg/`; `scripts/` launchers are in `scripts/` here).*

# Amendment 1 to prereg_cluster_revision.md: J6 (feature 2954 ablation type)

Frozen 2026-10-12, before any J6 number exists.

**Question.** Is 2954's +1.50 m z500 cost (zero-ablation vs its coverage/connectivity-matched
control f3404, `results/matched_controls.json`) an artefact of setting the feature to zero?

**Arms**, one invocation, 6 committed ICs (2020-01-10 … 2020-11-07), +48 h, global patch every step
(`global_rmse_ablate.py`, unchanged protocol): `baseline`, `f2954`/`ctrl_f2954` (zero, as committed),
`mean_f2954`/`ctrl_mean_f2954` (each feature replaced by its unconditional dataset mean,
mu = ssum / (160 × 40962) from `results/fs_feature_magnitude_stats.npy`; mu_2954 = 0.179, mu_3404 = 0.124).

**Bar.** z500 RMSE at +48 h, arm minus its control, paired over the 6 ICs.
- Not a zeroing artefact iff the mean-ablation difference is > 0 with paired p < 0.05 **and** is at
  least half the zero-ablation difference measured in the same invocation.
- If the mean difference falls below half the zero difference, or loses significance, report
  "2954's effect is substantially a zeroing artefact" in the paper.
- Cross-hardware: the zero-ablation difference should reproduce +1.50 m within ±0.3 m. If it
  does not, report that, and judge the bar only within this run.

**Not run (stated, not dropped):** resample ablation needs a node-varying patch from another IC's
codes, which `build_apply`'s (coef, rho, uvec) patch cannot express. Deferred.
