# `graphcast_sae.gridlock` — grid-locked features and what ablating them costs

Paper: Sec. 3, the grid-lock paragraph (0.1–1.5 m z500 at +48 h); demo notebook part 4.
Shipped outputs: `results/fs_gridlock_all.npy`, `results/matched_controls.json`,
`results/fs_global_rmse.npy`, `results/fs_footprints.npy`, `results/fs_footprints_extra.npy`,
`results/fs_footprint_rmse.npy`, `data/fs_footprint_fires_nw12.npz`.

Run order:

1. `gridlock_score_all.py` — Jaccard self-overlap of every feature's active node set across
   a year of i.i.d. windows → `fs_gridlock_all.npy` (CPU, needs the i.i.d. dump)
2. `footprint_inspect.py` / `footprints_extra.py` — footprints + connected components for the
   selected features and the convection group → `fs_footprints*.npy` (CPU)
3. `matched_control_draw.py` — a control per feature matched on coverage and connectivity
   → `matched_controls.json` (CPU)
4. `global_rmse_ablate.py` — global RMSE vs ERA5 under class / single-feature ablations,
   8 paired ICs × 120 h → `fs_global_rmse.npy` (GPU, ~hours)
5. `footprint_masks.py` → `footprint_rmse.py` → `footprint_rmse_analyze.py` — the same rollouts
   scored inside each footprint (GPU)
6. `rotation_all.py` (rotate the planet 180°: positional vs content features, 2 forwards),
   `rot_multidraw.py`, `f2075_climatology.py` — the positional tests
   `rotation_angles_summary.py` scores all five angles (45/90/135/180/270°; `scripts/rot_angles.sh`);
   `results/matched_controls_six.json` holds the controls for the six features that pass at every angle
   (`scripts/ablate_six.sh` runs their ablations; J6 adds mean ablation: `GR_ARMS` entries named `mean_*`, `scripts/rev_j6.sbatch`)

Environment: JAX env; steps 4–5 and `rotation_all.py` need the GPU.
