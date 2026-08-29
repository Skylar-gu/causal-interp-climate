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

Figures are numbered as in the paper. Figs 1-3 are web-rendered: each is an HTML page with its
data inlined (`figures/main_claims/figure<N>_web_notitle_print.html`), printed by headless
chromium and cropped (`CHROME_BIN=<chrome> python figures/main_claims/build_figure2p5.py` shows
the pipeline); `figures/main_claims/make_figures.py` is the matplotlib fallback for the same
three panels from the same shipped files.

| result | from shipped files | re-run (inputs → outputs) |
|---|---|---|
| Fig. 1 — SAVAR calibration, eastward-edge audit, propagation speed | `figures/main_claims/figure1_causal_discovery_notitle.pdf`; `python figures/main_claims/make_figures.py` (panel a from `savar/results/ladder_gnn/`) | SAVAR: `savar/README.md`; the audit and impulse numbers are read from the bundled values (Sec. 4 estimator lane not included) |
| Fig. 2 — interventions: (a) per-storm ablation of the spin feature f3316, (b) its dose–response, (c) medians for the three calibrated groups | `figures/main_claims/figure2p5_interventions_notitle.pdf` (`build_figure2p5.py` regenerates it from `results/skill/mech_3316`, `gain_3316`, `mech_spin3316`, `convection`, `moisture2`); the convection-triplet version of the same layout is `figure2_interventions_notitle.pdf`; notebook parts 2-3 | per arm: `storms.skill_conv_verify_era5` (CPU, WB2) → `MECH_RES=<arm> MECH_FEATS=<ids>` GPU `storms.skill_conv_run` (8 storms × 4 arms × 96 h, ~1 GPU-h) → `storms.skill_conv_analyze` → `verdict.json`. Shipped arms: `convection` 2401,2067,3174 · `mech_spin3316` 2089,2514,3316 · `mech_3316` 3316 · `mech_asc21` 553,866,1981 · `mech_shear` 2070,575,456,1514 · `mech_vort850` 2822,2935,1148,2089 (the polar, exposure-limited group an earlier draft called "low-level spin") · `moisture2` 2958,2671,37; random control 3667,2875,2850. Gain sweeps: `MECH_RES=gain_conv` / `gain_3316` with `MECH_GAINS=0,1.25,1.5,1.75,2,2.5,3` on `haishen2020 ida2021 patricia2015` (~25 GPU-min each) |
| Fig. 3 — ablation effects for three feature types | `figures/main_claims/figure3_gridlocked_notitle.pdf`; `make_figures.py` panel from `results/fs_footprints*.npy` | grid-lock lane below |
| Fig. 4 — recovered graph over feature groups with surrogate edges | `python figures/paper_fig_graphmap.py` (cartopy; `GRAPHMAP_FILL=color`) → `figures/paper_fig_graphmap_color.pdf` | pool + trajectory: `obsgraph.build_pool` → `build_pool_flag_v2` → `add_anchor_qrandc` (dump, ~15 min) → GPU `obsgraph.extract_traj_flag2 --start 2007-01-01 --n-steps 17532 --block 120 --dump-pooled $GC_SCRATCH/pooled` (~10 GPU-h) → `OMP_NUM_THREADS=8 obsgraph.finalize_traj_flag --pooled $GC_SCRATCH/pooled`. The PCMCI+ / matched-surrogate estimator that turns the series into `flag_gint.npy` and the residual-edge set is **not included**; the figure reads the bundled `fs_graphmap_inputs.npy` |
| Fig. 5 — Ida dial-up progression (calibrated groups) | `python figures/paper_fig_ida_dialup.py` (reads `results/fs_ida_mechmaps_prog_v2.npy`; `MECHMAPS_TAG=` for the original groups); notebook part 1 | GPU `MECHMAPS_TAG=_v2 MECHMAPS_GROUPS='{...}' storms.ida_mechmaps_prog` (5 × 48 h, ~4 GPU-min); groups from `storms.steer_ida_genesis_v2` (prereg `docs/prereg/prereg_ida_genesis_calibrated.md`), scored by `storms.ida_genesis_v2_analyze` → `results/fs_ida_genesis_v2_verdict.json`; single-feature decomposition `storms.steer_ida_genesis_v2_followup` |
| Fig. 6 — hurricane tracks under convection ablation / amplification | `figures/paper_fig_tracks.pdf` (shipped as-is) | `python figures/paper_fig_track.py` needs `results/skill/fields_conv/run_*.npy` (`MECH_RES=fields_conv MECH_FIELDS=1 MECH_GAINS=0,2` GPU `storms.skill_conv_run`), not shipped; `results/skill/era5_track.npy` is |
| Fig. 7 — removing grid-locked features, six features × seven fields | `figures/paper_fig_gridlocked_effects.pdf` (shipped as-is); notebook part 4 for the tables | `python figures/paper_fig_gridlocked_effects.py` (cartopy) reads `results/fs_matched_rmse.npy` (shipped) plus the rotation-angle scores and `hybrid_footprint_fires.npz` (not shipped). Lane: `gridlock.gridlock_score_all` (dump, ~30 min) → `footprint_inspect` / `footprints_extra` → `matched_control_draw` → GPU `gridlock.global_rmse_ablate` (8 ICs × 120 h per arm, several GPU-h) → `footprint_masks` → GPU `footprint_rmse` → `footprint_rmse_analyze`; positional tests `gridlock.rotation_all` |

The Ida genesis knockout that selects Fig. 5's groups and gives the internal-readout numbers
(convection −41 %, low-level spin −54 %, moisture +3 %, shear +3 %, all four −80 %; f3316 alone
−45 %) is `results/fs_ida_genesis_v2.npy` / `_followup.npy` / `_verdict.json`, with the record in
`docs/notes/result_ida_genesis_calibrated_2026_08_29.md`.

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
| `figures/paper_fig_tracks.pdf`, `figures/paper_fig_gridlocked_effects.pdf` | Figs 6 and 7, shipped as rendered (their `fields_conv` runs, rotation-angle scores and hybrid fires are not included) | — |

Not shipped and not written here (each script's `Inputs:` line flags them): `results/flag_gint*`,
`flag_signature_physics.npy`, `litext_gc_*` (the Sec. 4 estimator lane); `hybrid_footprint_fires.npz`;
`fs_pcmci_nodes.npy`; `fs_ida_{castsel,mechfeats,trop}.npy`; `fs_deleted_norm.npy`;
`fs_concept_rmse.npy`; the `legacy/` steering outputs. Where a script needs one, the paper number
it supports is read from a shipped verdict or bundle instead.
