# Reproducing the paper's results

Every figure and table in the paper regenerates on CPU in minutes from the shipped
`results/`, `data/` and `results/*_inputs.npy` bundles, via `figures/paper_fig_*.py` and
`notebooks/demo.ipynb`. Re-running the experiments themselves needs the DeepMind GraphCast
checkpoint, WeatherBench2 ERA5 (streamed from `gs://weatherbench2/...`), one ~46 GB GPU,
and for most CPU analyses the 6.7 GB i.i.d. activation dump. A few curated inputs come
from an analysis pipeline that is not included; they are listed at the end.

## Environment

* Demo, figures and CPU analyses: Python 3.9, `pip install -r requirements.txt`
  (`requirements-dev.txt` adds `pytest` and the notebook runner).
* GPU experiments ("JAX env"): Python 3.11 with `jax[cuda12]`, `dm-haiku`,
  `graphcast @ git+https://github.com/google-deepmind/graphcast.git`, `xarray`, `zarr`,
  `gcsfs`; versions used are in the GPU block of `requirements.txt`. Three CPU catalog
  scripts (`atlas/feature_select`, `obsgraph/build_pool`, `extraction/fs_catalog`) need `torch`.
* Locations are resolved in `graphcast_sae/paths.py`: `GRAPHCAST_PARAMS` (checkpoint),
  `GRAPHCAST_ASSETS` (its `stats/`), `GC_SCRATCH` (regenerable dumps; default `scratch/`).
* Run scripts from the repository root as `python -m graphcast_sae.<group>.<script>`; every
  script's docstring ends with `Paper / Inputs / Outputs / Run`. `python -m pytest tests -q`
  checks imports and path hygiene.
* The i.i.d. dump: `FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -m graphcast_sae.extraction.extract_iid_dump --n 160`
  → `$GC_SCRATCH/fs_iid_dump.npy` (160 windows × 40,962 nodes × 512, fp16) + `fs_iid_meta.json`.

`GPU` below means `FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -m …` in the JAX env.

## Main-text results

| result | from shipped files | re-run (inputs → outputs) |
|---|---|---|
| Fig. `fig:savar` — SAVAR calibration | `python figures/paper_fig_savar.py` | self-contained in `savar/README.md` |
| Fig. `fig:contrast` — generic-state dose vs the convection lever | `python figures/paper_fig_contrast.py` (cartopy) | `concepts.cgv2_actseries` → `cgv2_select` → GPU `concepts.respop` (23 rolls × 60 h × 4 windows, several GPU-h) → `concepts.respop_score` → `results/fs_respop.npy` (51 MB, bundled as `fs_contrast_inputs.npy`); needs the dump + `atlas.label_expanded`/`label_extra` |
| Table `tab:mechanism-interventions` — seven-storm ablations | `notebooks/demo.ipynb` part 2 (`results/skill/<arm>/verdict.json`) | per arm: `storms.skill_conv_verify_era5` (CPU, WB2) → `MECH_RES=<arm> MECH_FEATS=<ids>` GPU `storms.skill_conv_run` (8 storms × 4 arms × 96 h, ~1 GPU-h) → `storms.skill_conv_analyze` → `verdict.json`. Shipped arms: `convection` 2401,2067,3174 · `mech_asc21` 553,866,1981 · `mech_shear` 2070,575,456,1514 · `mech_vort850` 2822,2935,1148,2089 · `moisture2` 2958,2671,37; random control 3667,2875,2850. Controls: `storms.inbox_control`, `storms.core_scan` → `core_control_all` (dump) |
| Fig. `fig:gain` — dose–response | `python figures/paper_fig_gain.py` | `MECH_RES=gain_conv MECH_GAINS=0,1.25,1.5,1.75,2,2.5,3` GPU `storms.skill_conv_run` (~1 GPU-h) → `storms.gain_accuracy` → `results/skill/gain_conv/` |
| Fig. 5 — Ida dial-up progression | `python figures/paper_fig_ida_dialup.py`; notebook part 1 | GPU `storms.ida_mechmaps_prog` (5 × 48 h, ~20 GPU-min) → `results/fs_ida_mechmaps_prog.npy`; feature groups from the curated `fs_ida_mechfeats.npy`, cast from `storms.ida_scan` |
| Grid-locked-feature ablations (Sec. 3) | `notebooks/demo.ipynb` part 4; `python figures/paper_fig_maps.py` (cartopy) | `gridlock.gridlock_score_all` (dump, ~30 min) → `footprint_inspect` / `footprints_extra` → `matched_control_draw` → GPU `gridlock.global_rmse_ablate` (8 ICs × 120 h per arm, several GPU-h) → `footprint_masks` → GPU `footprint_rmse` → `footprint_rmse_analyze`; positional tests `gridlock.rotation_all` |
| Fig. `fig:graphmap` — observational consensus graph (Sec. 4) | `python figures/paper_fig_graphmap.py` (cartopy; `GRAPHMAP_FILL=color` shaded) | pool + trajectory: `obsgraph.build_pool` → `build_pool_flag_v2` → `add_anchor_qrandc` (dump, ~15 min) → GPU `obsgraph.extract_traj_flag2 --start 2007-01-01 --n-steps 17532 --block 120 --dump-pooled $GC_SCRATCH/pooled` (~10 GPU-h) → `OMP_NUM_THREADS=8 obsgraph.finalize_traj_flag --pooled $GC_SCRATCH/pooled`. The PCMCI+ / matched-surrogate estimator that turns the series into `flag_gint.npy` and the residual-edge set is **not included**; the figure reads the bundled `fs_graphmap_inputs.npy` |

Appendix experiments (parity family, sparsity-budget competition and mediation, receptive field
and local aggregate, hybrid PCMCI+ null, skill decomposition) live in `graphcast_sae/appendix/`
and `graphcast_sae/obsgraph/`; each group's `README.md` lists its scripts in run order and each
script's docstring gives the exact command. Pre-registrations are in `docs/prereg/`.

## Curated inputs shipped as-is

Read by scripts or figure builders here, produced by the analysis pipeline that is not included:

| file | what it is | read by |
|---|---|---|
| `results/fs_graphmap_inputs.npy` | consensus graph (`flag_gint.npy`, `leiden_flag` member) + residual-edge set | `paper_fig_graphmap.py` |
| `results/fs_contrast_inputs.npy` | bundle of the RESPOP run (regenerable, see above) | `paper_fig_contrast.py` |
| `results/fields8_track.npy` | per-storm field snapshots | `paper_fig_replic.py` |
| `results/fs_global_rmse_topamp.npy`, `fs_perfeat_rmse.npy`, `fs_feature_magnitude_stats.npy` | per-feature ablation summaries of the grid-lock lane | `notebooks/demo.ipynb` |
| `results/land_sea_mask_025.npz` | 0.25° land-sea mask | `paper_fig_ida_dialup.py` |
| `data/fs_footprint_fires_nw12.npz`, `data/mesh_2to6_geom.npy` | per-feature firing footprints over 12 dates; M6 mesh geometry | figures, `gridlock/`, `obsgraph/` |
| `results/heatdome/physics_verdict.json` | heat-dome verdict (regenerable via `heatdome/`; not in the paper) | notebook |

Not shipped and not written here (each script's `Inputs:` line flags them): `results/flag_gint*`,
`flag_signature_physics.npy`, `litext_gc_*` (the Sec. 4 estimator lane); `hybrid_footprint_fires.npz`;
`fs_pcmci_nodes.npy`; `fs_ida_{castsel,mechfeats,trop}.npy`; `fs_deleted_norm.npy`;
`fs_concept_rmse.npy`; the `legacy/` steering outputs. Where a script needs one, the paper number
it supports is read from a shipped verdict or bundle instead.
