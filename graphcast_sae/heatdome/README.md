# `graphcast_sae.heatdome` — the 2021 Pacific-Northwest heat dome (not in the paper)

A blocking-necessity study on the June 2021 omega block: is the model's "ridge" code a
small causal lever the way convection is for cyclones? It is not in the paper; it ships
because `results/heatdome/physics_verdict.json` is read by the demo notebook and by
`figures/paper_fig_intervention.py`, and the answer (blocking is distributed) is quoted
in the discussion of selective leverage.

Run order (JAX env, GPU, 6-day rollouts from IC 2021-06-24; `heatdome_config.py` is frozen):

1. `heatdome_verify_era5.py` — ERA5 ridge + heat truth for the W-NA box (CPU)
2. `heatdome_phase1.py` — which features fire on the ridge → `results/heatdome/phase1.npy` (regenerable, not shipped)
3. `heatdome_phase2.py` — restore-to-normal knockout of those features → `phase2.npy`;
   `heatdome_analyze.py` → `results/heatdome/verdict.json` (regenerable, not shipped)
4. `heatdome_scan.py` (all 4096 features on the ridge) → `heatdome_scan_analyze.py`
   (physics-guided sets) → `heatdome_physics_ablate.py` → `heatdome_physics_analyze.py`
   → `results/heatdome/physics_verdict.json` (shipped)
5. `heatdome_topk_ablate.py` — nested top-k dose response

Only `physics_verdict.json` ships; the `.npy` intermediates are regenerable.
