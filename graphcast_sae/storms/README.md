# `graphcast_sae.storms` — mechanism ablations on seven tropical cyclones, Ida dial-up, gain sweep

Paper: Sec. 3 "The intervention contrast" — Table `tab:mechanism-interventions`,
Fig. `fig:gain`, Fig. 5 (Ida dial-up). Shipped outputs: `results/skill/<arm>/`
(`run_<storm>.npy`, `era5_truth.npy`, `verdict.json`), `results/skill/era5_track.npy`,
`results/fs_ida_mechmaps_prog.npy`, `results/skill/convection/ps5_displacement.json`.

Run order for one arm (JAX env; the rollouts need a ~46 GB GPU, ~16 rollouts × 96 h per arm):

1. `skill_conv_verify_era5.py` — ERA5 truth per storm (CPU, streams WB2) → `era5_truth.npy`
2. `skill_conv_run.py` — four-arm counterfactual rollouts sharing one compiled graph
   (`MECH_RES=<arm>`, `MECH_FEATS=<ids>`, `MECH_GAINS=…` for the gain sweep) → `run_<storm>.npy`
3. `skill_conv_analyze.py` — deepening lost / skill deltas → `verdict.json`
4. `era5_track.py` — ERA5 MSLP track per storm (for `figures/paper_fig_track.py`)

Controls and audits: `inbox_control.py` (in-box-matched random control), `core_scan.py` →
`core_control_all.py` (core-activation-matched control), `core_firing_vs_effect.py`,
`conv_radius_sweep_analyze.py` (1500 km disk), `commit_horizon_analyze.py` (single-step pulses),
`hres_compare.py` (GraphCast vs IFS-HRES), `gain_accuracy.py` / `gain_global_rmse.py` /
`storm_character.py` (dose–response), `ps5_shear_displacement.py` (downshear test),
`event_screen.py`, `xt_locate.py`, `locate_sh_storms.py`, `skill_conv_actdump.py` (extra batteries).

Ida dial-up (Fig. 5): `ida_scan.py` (the local cast) → `steer_ida_counterfactual.py` (the
restore-to-normal counterfactual every arm reuses) → `ida_mechmaps_prog.py` →
`results/fs_ida_mechmaps_prog.npy` → `figures/paper_fig_ida_dialup.py`.
