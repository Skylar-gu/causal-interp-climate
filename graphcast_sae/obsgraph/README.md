# `graphcast_sae.obsgraph` — the observational graph, audited

Paper: Sec. 4 (Fig. `fig:graphmap`) and Appendix `app:null`. Shipped output:
`results/fs_graphmap_inputs.npy` (the bundle `figures/paper_fig_graphmap.py` reads:
consensus graph, footprints, the 24 residual edges, `frac_eastward`).

Run order (the flag_gint chain; JAX env, GPU for step 3):

1. `build_pool.py` → `build_pool_flag_v2.py` → `add_anchor_qrandc.py` — the pool: Leiden /
   varimax / k-means / SAE decompositions of the i.i.d. dump plus the two corrupted anchors
   (`discover_leiden.py` is the community-detection backend) → `candidates/pool_flag_v2_*.npy`
2. `extract_traj_flag2.py` — 12-year contiguous teacher-forced trajectory projected through
   every pool member (`--dump-pooled $GC_SCRATCH/pooled`) → mode series
3. `finalize_traj_flag.py` — zero-GPU anchors + the pre-registered data gate
   (`docs/prereg/prereg_flagship_gint.md`)
4. PCMCI+ on the series and the matched-surrogate ladder produce `results/flag_gint.npy` (not shipped)
   (**not included here**, see `docs/REPRODUCE.md`); `report_flag_gint.py` prints the FG-6
   table and `physics_verdict.py` the storm-track verdict from it

Hybrid design (Appendix `app:null`): `extract_concept_traj.py` (concept series) →
`hybrid_pcmci.py select | condition | graph | nullgraph` (CPU) → `hybrid_score.py`;
`hybrid_calibrate_windows.py` is the consensus-vs-joint power argument.
