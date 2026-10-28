# Reproducing the paper

Figures and tables regenerate on CPU from the shipped `results/`, except where a row names a file
as not shipped. The GPU column says how each result was produced (one ~46 GB GPU; see the README
for the external assets). Commands run from the repository root.

| paper | regenerate (CPU) | produced by (GPU) |
|---|---|---|
| Fig. 1 — grid-locked, land-locked and vorticity footprints with $z_{500}$ ablation scores | `figures/fig1_gridlocked.html` is self-contained (data inlined); `figures/fig1_gridlocked.pdf` is its print | `graphcast_sae.gridlock.global_rmse_ablate` via `scripts/ablate_six.sh` → `results/fs_matched_rmse_six.npy` |
| Fig. 2 — (a) restore-to-normal of 2401 and 3316; (b, c) gain sweeps on Ida and Haishen | `python figures/fig2_interventions.py` (playwright for the PDF) reads `results/skill/mech_2401`, `mech_3316`, `fig2_2401`, `gain_3316` | `graphcast_sae.storms.skill_conv_run` (`MECH_RES=<arm> MECH_FEATS=<ids>`), scored by `graphcast_sae.storms.skill_conv_analyze`; sweeps via `scripts/rev_fig2.sbatch`, controls from `graphcast_sae.storms.core_control_single` |
| Fig. 3 — Ida cyclone feature under doubled convection / vorticity groups, ERA5-forced row | `python figures/fig3_ida_dialup.py` reads `results/fs_ida_mechmaps_prog_v2.npy` | `graphcast_sae.storms.ida_mechmaps_prog` (`MECHMAPS_TAG=_v2`); the ERA5-forced row (`tc_truth`, the 213) comes from `graphcast_sae.storms.ida_mechmaps_truth` and is not shipped; groups from `graphcast_sae.storms.steer_ida_genesis_v2` and `ida_genesis_v2_analyze` |
| Fig. 4, Fig. 5 — grid-locked effects across fields and as a $z_{500}$ map | `python figures/fig4_gridlocked_effects.py`, `python figures/fig5_gridlocked_z500map.py` (cartopy); they also read `results/hybrid_footprint_fires.npz` and `results/gridlock_z500_perfeat_field.npy` (not shipped), so the shipped PDFs are the paper's renders | `graphcast_sae.gridlock.gridlock_score_all` → `matched_control_draw` → `global_rmse_ablate`; rotations `graphcast_sae.gridlock.rotation_all` (`scripts/rot_angles.sh`), criterion `rotation_angles_summary` |
| Table 1, Table 3, abstract and Sec. 3.2 percentages | `python -m graphcast_sae.storms.single_feature_tables`; bias correction `python -m graphcast_sae.storms.offset_baseline` | J8 sweeps (prereg amendments 4–6), scored by `graphcast_sae.storms.rev_analyze`; `results/skill/single_feature_mslp.json` was transcribed from the cluster runs (rollouts not shipped) |
| Table 2 — Southern Hemisphere | `MECH_RES=sh_suff_ctl python -m graphcast_sae.storms.rev_analyze j1` | `scripts/rev_j7.sbatch` |
| Restore-to-normal medians (2.5 / 3.0 hPa) | `results/skill/mech_2401/verdict.json`, `results/skill/mech_3316/verdict.json`, seven developing storms | as Fig. 2a |
| App. A labels | `results/fs_mechanisms_v2.npy` | `graphcast_sae.atlas.label_mechanisms_v2` on the i.i.d. dump (`graphcast_sae.extraction.extract_iid_dump`) |
| 1,500 km disk, saturation by 750 km | `python -m graphcast_sae.storms.conv_radius_sweep_analyze` → `results/skill/conv_radius_sweep.json` | `MECH_RES=conv_r<R>` storm runs (not shipped) |
| Random-direction controls, run-to-run floor | `results/skill/dirs_j2`, `results/skill/determinism_j4` | `scripts/rev_j2.sbatch`, `scripts/rev_j4.sbatch` |

Cluster setup: `GC_SCRATCH=<scratch> bash scripts/setup_rorqual.sh`, then `sbatch` a launcher;
`scripts/rev_env.sh` holds the shared environment. Pre-registrations are in `docs/prereg/`.
