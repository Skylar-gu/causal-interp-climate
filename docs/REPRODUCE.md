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

The current draft has two result sections — grid-locked features (Fig. 1) and tropical-cyclone
interventions (Figs. 2–3) — plus two appendix figures and one appendix table. The
`figures/main_claims/*_notitle.pdf` files keep their historical numbers in the **filename**
(`figure3_…` is paper Fig. 1, `figure2_…` is paper Fig. 2); each is an HTML page with its data
inlined (`figure<N>_web_notitle_print.html`), printed by headless chromium and cropped
(`CHROME_BIN=<chrome> python figures/main_claims/build_figure2p5.py` shows the pipeline).
`figures/main_claims/make_figures.py` is the matplotlib fallback for the same panels from the
same shipped files.

| result | from shipped files | re-run (inputs → outputs) |
|---|---|---|
| Fig. 1 — grid-locked features are causally relevant: footprints of a grid-locked, a land-locked and a convection feature, with global $z_{500}$ ablation scores | `figures/main_claims/figure3_gridlocked_notitle.pdf` (paper Fig. 1); `make_figures.py` panel from `results/fs_footprints*.npy` | grid-lock lane below |
| Fig. 2 — interventions (camera-ready): convection group, seven storms, every gain against an in-box control at the same gain: (a) deepening removed at α=0, (b) deepening-error reduction vs ERA5 per storm, (c) predicted deepening as a share of ERA5's | `figures/main_claims/fig2_interventions.pdf` via `python figures/main_claims/fig2_interventions.py` from `results/skill/suff_ctl/rev_verdict.json` (J1, below). The submitted draft's version is `figure2_interventions_notitle.pdf` (`convection` (a), `gain_conv` (b)); the single-spin-feature layout is `figure2p5_interventions_notitle.pdf` via `figures/main_claims/build_figure2p5.py` | J1 below |
| Fig. 2 (submitted draft) | `figures/main_claims/figure2_interventions_notitle.pdf` (paper Fig. 2: `convection` (a), `gain_conv` (b)); the single-spin-feature layout is `figure2p5_interventions_notitle.pdf` via `build_figure2p5.py` (`results/skill/mech_3316`, `gain_3316`, `mech_spin3316`, `convection`, `moisture2`); notebook parts 2-3 | per arm: `storms.skill_conv_verify_era5` (CPU, WB2) → `MECH_RES=<arm> MECH_FEATS=<ids>` GPU `storms.skill_conv_run` (8 storms × 4 arms × 96 h, ~1 GPU-h) → `storms.skill_conv_analyze` → `verdict.json`. Shipped arms: `convection` 2401,2067,3174 · `mech_2401` 2401 · `mech_spin3316` 2089,2514,3316 · `mech_3316` 3316 · `mech_asc21` 553,866,1981 · `mech_shear` 2070,575,456,1514 · `mech_vort850` 2822,2935,1148,2089 (the polar, exposure-limited group an earlier draft called "low-level spin") · `moisture2` 2958,2671,37; random control 3667,2875,2850. Gain sweeps: `MECH_RES=gain_conv` / `gain_3316` with `MECH_GAINS=0,1.25,1.5,1.75,2,2.5,3` on `haishen2020 ida2021 patricia2015` (~25 GPU-min each); `gain_3316_ext` extends Ida to α=5 (`MECH_GAINS=3.5,4,4.5,5`, prereg amendment 1) |
| Fig. 3 — Ida dial-up progression: the internal cyclone feature f3243 at +12/+24/+36/+48 h, baseline vs each mechanism group doubled, with an ERA5-forced reference row | `figures/paper_fig_ida_dialup.py` → `paper_fig_ida_dialup.pdf` (reads `results/fs_ida_mechmaps_prog_v2.npy`; `MECHMAPS_TAG=` for the original groups); notebook part 1. The paper renders the reduced convection+vorticity+ERA5-forced layout | GPU `MECHMAPS_TAG=_v2 MECHMAPS_GROUPS='{...}' storms.ida_mechmaps_prog` (5 × 48 h, ~4 GPU-min); groups from `storms.steer_ida_genesis_v2` (prereg `docs/prereg/prereg_ida_genesis_calibrated.md`), scored by `storms.ida_genesis_v2_analyze` → `results/fs_ida_genesis_v2_verdict.json`; single-feature decomposition `storms.steer_ida_genesis_v2_followup` |
| Tab. 1 — single-feature domination of each intervention group (median deepening removed, feature alone vs full group) | `results/skill/{convection,mech_2401,mech_spin3316,mech_3316}/verdict.json`; notebook part 2 | skill_conv lane above; `mech_2401` (prereg `docs/prereg/prereg_mech_2401.md`) is the single convection feature, `mech_3316` the single spin feature |

## Camera-ready revision runs (J1–J7)

Run on an H100 cluster after review, under `docs/prereg/prereg_cluster_revision.md` (frozen
before any number) and amendments 1–3; the result record is
`docs/notes/result_cluster_revision_2026_10_12.md`, the plan `docs/notes/spec_cluster_port.md`.
SLURM launchers are in `scripts/` (`rev_env.sh` + `rev_*.sbatch`; submit from the repository
root, `mkdir -p out` once). All but J4 use `XLA_FLAGS=--xla_gpu_deterministic_ops=true`
(amendment 3); arms are only ever differenced within one run.

| job | what | launcher → scorer | shipped result |
|---|---|---|---|
| J1 (+J3) | convection gain sweep α ∈ {0, 0.5, 1, 1.25, 1.5, 2, 2.5, 3} on 7 storms + non-developer, in-box matched control at every gain; TC feature 3243 read before the patch (J3) | `rev_j1.sbatch` (GPU `storms.skill_conv_run`) → `MECH_RES=suff_ctl storms.rev_analyze j1` | `results/skill/suff_ctl/rev_verdict.json`, `rev_analysis.txt` (paper Fig. 2, Tab. 1) |
| J2 | random-direction and difference-of-means baselines at matched per-node magnitude | `rev_j2.sbatch` | `results/skill/dirs_j2/` |
| J4 | run-to-run floor: 5 copies of two arms, default XLA | `rev_j4.sbatch` → `storms.rev_analyze j4` | `results/skill/determinism_j4/rev_verdict.json` (0.25 hPa) |
| J6 | feature 2954: zero vs mean ablation, 6 ICs | `rev_j6.sbatch` (GPU `gridlock.global_rmse_ablate`) | `results/fs_j6_ablation_type.npy`, `j6_status.txt`, `j6_groups.json` |
| J7 | J1 on five Southern Hemisphere storms | `rev_j7.sbatch` → `SKILL_STORMS=skill_sh_storms MECH_RES=sh_suff_ctl storms.rev_analyze j1` | `results/skill/sh_suff_ctl/` |
| gate 2 | zero-patch identity under default vs deterministic XLA | `rev_gate2_det.sbatch` | `results/skill/suff_ctl_nondet/` (default XLA, failed; its `err_red`/`spec` fields were scored before the metric fix and hold the min-MSLP RMSE metric) |

Compute nodes without internet read ERA5 from a sparse local copy:
`python -m graphcast_sae.extraction.stage_era5_local --out $GC_SCRATCH/era5_wb2.zarr`
on a node with internet, then `FS_ERA5_LOCAL` (set in `rev_env.sh`). Exact environments are
in `env/`.

The same pass shipped the inputs behind the paper's other camera-ready numbers:
grid-lock rotation scores at five angles (`results/fs_rotation_all*.npy`,
`gridlock.rotation_angles_summary`, `docs/notes/rotation_angles_2026_08_25.md`), the six
grid-locked features against matched controls (`results/fs_matched_rmse_six.npy`,
`matched_controls_six.json`, `gridlock.matched_control_draw_six`, `scripts/ablate_six.sh`),
the 16 concept-group restore-to-normal batteries (`results/skill/mech_*`, `moisture`,
`docs/notes/mechanism_library_2026_08_17.md`), core activation and core-matched controls
(`results/fs_core_scan.npy`, `core_control*.json`, `results/skill/conv_corectl{2,4}`,
`storms.core_control_all`), the in-box controls (`results/inbox_control_*.json`,
`results/skill/convection_inboxctl`), the disk-radius sweep (`results/skill/conv_radius_sweep.json`)
and the HRES / best-track comparison (`results/hres_compare.npy`, `docs/notes/result_hres_gap.md`).

App. Fig. 4 (grid-locked effects, six features × seven fields) and App. Fig. 5 (the same
features as a per-gridpoint $z_{500}$ error map) are in the appendix table below.

The Ida genesis knockout that selects Fig. 3's groups and gives the internal-readout numbers
(convection −41 %, low-level spin −54 %, moisture +3 %, shear +3 %, all four −80 %; f3316 alone
−45 %) is `results/fs_ida_genesis_v2.npy` / `_followup.npy` / `_verdict.json`, with the record in
`docs/notes/result_ida_genesis_calibrated_2026_08_29.md`.

**Not in the current draft.** Earlier drafts also carried a SAVAR calibration figure
(`figures/main_claims/figure1_causal_discovery_notitle.pdf`, `savar/`), the observational
PCMCI+ graph over feature groups (`figures/paper_fig_graphmap.py`, `graphcast_sae/obsgraph/`,
`results/fs_graphmap_inputs.npy`) and hurricane tracks under convection ablation
(`figures/paper_fig_tracks.pdf`, `figures/paper_fig_track.py`). Their code and shipped result
bundles are retained but no longer referenced by the paper, the same status as
`graphcast_sae/heatdome/`.

Appendix experiments (parity family, sparsity-budget competition and mediation, receptive field
and local aggregate, hybrid PCMCI+ null, skill decomposition) live in `graphcast_sae/appendix/`
and `graphcast_sae/obsgraph/`; each group's `README.md` lists its scripts in run order and each
script's docstring gives the exact command. Pre-registrations are in `docs/prereg/`.

| appendix figure | from shipped files | re-run |
|---|---|---|
| App. Fig. 4 — removing grid-locked features, six features × seven fields | `figures/paper_fig_gridlocked_effects.pdf` (shipped as rendered); notebook part 4 for the tables | `python figures/paper_fig_gridlocked_effects.py` (cartopy) reads `results/fs_matched_rmse.npy` (shipped) plus the rotation-angle scores and `hybrid_footprint_fires.npz` (not shipped). Lane: `gridlock.gridlock_score_all` (dump, ~30 min) → `footprint_inspect` / `footprints_extra` → `matched_control_draw` → GPU `gridlock.global_rmse_ablate` (8 ICs × 120 h per arm, several GPU-h) → `footprint_masks` → GPU `footprint_rmse` → `footprint_rmse_analyze`; positional tests `gridlock.rotation_all` |
| App. Fig. 5 — grid-locked features as a per-gridpoint $z_{500}$ error map (+48 h), one shared colour scale | `figures/paper_fig_gridlocked_effects_z500map.pdf` (shipped as rendered) | `python figures/paper_fig_gridlocked_effects_z500map.py` (cartopy) needs `results/gridlock_z500_perfeat_field.npy` (per-gridpoint field, ~0.35 GB, **not shipped**), `results/hybrid_footprint_fires.npz` and the rotation-angle scores (not shipped); `data/mesh_2to6_geom.npy` and `data/fs_footprint_fires_nw12.npz` ship |

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
| `figures/paper_fig_gridlocked_effects.pdf`, `figures/paper_fig_gridlocked_effects_z500map.pdf` | App. Figs. 4 and 5, shipped as rendered (rotation-angle scores, `hybrid_footprint_fires.npz` and the per-gridpoint field are not included) | — |
| `figures/paper_fig_tracks.pdf` | hurricane tracks, shipped as rendered; not in the current draft (`fields_conv` runs not included) | — |

Not shipped and not written here (each script's `Inputs:` line flags them): `results/flag_gint*`,
`flag_signature_physics.npy`, `litext_gc_*` (the Sec. 4 estimator lane); `hybrid_footprint_fires.npz`;
`fs_pcmci_nodes.npy`; `fs_ida_{castsel,mechfeats,trop}.npy`; `fs_deleted_norm.npy`;
`fs_concept_rmse.npy`; the `legacy/` steering outputs. Where a script needs one, the paper number
it supports is read from a shipped verdict or bundle instead.
