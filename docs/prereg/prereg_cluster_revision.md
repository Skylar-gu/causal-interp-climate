*Historical document, kept as written; script paths refer to the camera-ready `causal-graphcast` layout (`src/` and `flagship_sae/` are now `graphcast_sae/`, regrouped by experiment; `notes/` is now `docs/notes/` and `docs/prereg/`; `scripts/` launchers are in `scripts/` here).*

# Prereg: Paper 199 revision runs on Rorqual (J1, J3, J4, J7; J2/J6 to follow)

Frozen 2026-10-12, **before any cluster number exists**. Bars are taken verbatim from
`notes/spec_cluster_port.md` §6–7 (commit 3ba02ec); this file only adds what the port forced.
Any later change is an amendment file, never an edit here.

## Hardware / provenance (deviations from the L40S box, declared up front)

- GPU: NVIDIA H100 80GB (Rorqual `gpubase_*`), not the L40S. bf16, `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- Env: `.venv-jax` rebuilt from `env/requirements-jax-freeze.txt` (jax 0.10.2, graphcast @ 08cf736,
  numpy 2.4.6) on miniforge Python 3.11.17, with Alliance's `PYTHONPATH` unset. **Cartopy omitted**
  (failed to build; not imported by any job here). The spec's "freeze the flagship env on the
  L40S box" step was never done, so the ground-truth env cannot be diffed; gate 3 below is the check.
- Weights: md5 match for checkpoint, SAE `.pt`, `config.json`. The derived SAE `.npz` was rebuilt
  from the `.pt` (md5 `f58b5e75…`, spec lists `b0b92a17…`; same arrays expected, different zip bytes).
- ERA5: sparse byte-identical chunk copy of the WB2 zarr (`stage_era5_local.py`), read via
  `FS_ERA5_LOCAL`; compute nodes have no internet.
- **No i.i.d. dump on this cluster** (`fs_iid_dump.npy` lives on the L40S box / S3). The J1
  off-manifold flag is therefore recorded as raw per-step in-disk quantiles of the steered
  convection codes (q99, q99.9, max) per arm, to be compared with the i.i.d. 99.9th percentile
  once the dump is available. No off-manifold claim is made from this run alone.
- Every verdict carries `provenance = {gpu, driver, cuda, jax, cluster}`. No arm from this cluster
  is ever differenced against an arm from the L40S box.

## Sanity gates (stop on any failure; reported as instrument failures if they fail)

1. `jax.devices()` lists a GPU; md5s as above.
2. Zero-patch identity inside one graph: `gain-1` == `baseline`, min-MSLP bit-for-bit at every lead.
3. Cross-hardware reproduction: Ida `gain-0` (== conv-normal) `d_deepen` vs this run's baseline
   = 2.68 hPa (`convection_inboxctl`) within the p90 floor, **0.37 hPa**.
4. Merge gate: when combining runs, the baseline arm must agree within the floor or the merge is refused.

## J1 — controlled gain sweep (+ J3 readout, folded in)

Arms per storm, all in ONE invocation: `baseline`, `gain-g` and `rand-gain-g` for
g ∈ {0, 0.5, 1, 1.25, 1.5, 2, 2.5, 3}, plus `rand-normal`; in-box matched control from
`results/inbox_control_skill_conv_storms.json`; `MECH_FIELDS=1`.

- **Specific sufficiency** is claimed iff (treatment − in-box control) deepening-error reduction at
  g = 2 exceeds the J4 floor in **≥ 5 of 7** developing storms. Patricia is pre-declared under-resolved.
- **Guardrail #9:** the control must be able to fail. In-box firing 87–112% of the group (already
  true). Haiyan, where the control already costs 2.8 hPa, is the pre-declared demonstration that it
  is not inert. If no storm shows a control effect beyond the floor at any gain, the control is
  reported as possibly inert and the sufficiency claim is **not** made.
- Every gain is reported; g = 2 is in-sample and labelled so.
- If J4 has not landed, the floor used is the inferred p90, 0.37 hPa, and the result is labelled provisional.

## J3 — TC-feature readout

Feature 3243 is reported from the **pre-patch** activation (`X_pre`, next step), next to the old
post-patch value. The post − pre gap is the leak. An increase counts only if it exceeds the in-box
control's 3243 change at the same gain. The Fig 3 headline uses pre-patch only.

## J4 — determinism replicate

Ida and Patricia; `baseline` ×5 and `conv-normal` (gain-0) ×5 as separate arms in one invocation.
Floor = max pairwise |Δ min-MSLP| and |Δ d_deepen| across the 5 copies, per arm. This replaces the
inferred 0.15/0.37 hPa. No 0.1–0.6 hPa number is quoted before it exists.

## J7 — Southern Hemisphere gain sweep

Same as J1 with `SKILL_STORMS=skill_sh_storms`, `MECH_ALLOW_OVERLAP=1`, SH in-box controls. Same bar
(≥ 4 of 5 developing SH storms at g = 2, scaled from 5/7), reported whichever way it falls.

## False alarm

Non-developer deepening > J4 floor at any g ≤ 2 is reported as spurious spin-up.

## Not in this prereg

J2 (direction baselines), J5 (80-storm battery), J6 (2954 ablation types) get their own amendment
before their analysis code is written.
