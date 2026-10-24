*Historical document, kept as written; script paths refer to the camera-ready `causal-graphcast` layout (`src/` and `flagship_sae/` are now `graphcast_sae/`, regrouped by experiment; `notes/` is now `docs/notes/` and `docs/prereg/`; `scripts/` launchers are in `scripts/` here).*

# Spec: running the Paper 199 revision experiments on another cluster

v2, 2026-10-12. Supersedes v1 (commit 431cd9b), which was written from a clone that had not
fetched the 89 commits of 2026-08-14 → 08-25. Those commits already ran about half of what v1
planned; section 1 lists them so nobody re-runs them.

Companion to the revision spec (https://claude.ai/artifact/PHr1w4r6iTFGatLUnkZL6Q), which
ranks *what* to do. This note says *how to run the remaining GPU work on a cluster other than
the L40S box*. Read `CLAUDE.md` first; its rules apply unchanged on any cluster.

---

## 0. TL;DR

1. **Already done, don't rerun (section 1).** This covers:
   - the Eq. 1 dose-response on 7 storms (`delta_gain`, `suff_convection`);
   - the in-box matched control (`convection_inboxctl`) and the core-matched control (`conv_corectl4`);
   - the 0.15 hPa noise floor;
   - the HRES and best-track gap;
   - the 6-feature grid-lock ablations with matched controls;
   - rotation scoring over all 4,096 features;
   - the 750 km radius sweep;
   - an 80-storm registry, gated but not yet run.
2. **Still needs GPU (section 5):** seven jobs, J1–J7, ≈ 70–110 GPU-h in total. Most of
   that is J5, the 80-storm battery. **J1, J2 and J7 together are ≈ 12 GPU-h and answer the
   review.**
3. **Environment:** freeze `.venv-jax` on the L40S box *before* leaving it. That env is the
   one every committed number ran in (`CLAUDE.md` → Practical facts).
4. **Assets:** about 200 MB of weights with md5s (section 3), plus ERA5 from WB2. If the
   compute nodes have no internet access, J5 alone needs about 80 storm windows staged.
5. **Paths:** `fs_common.ROOT` and `SCRATCH` are hard-coded to `/home/ec2-user`. Section 4
   has the patch.
6. **The one rule that bites on a cluster.** Every arm you intend to compare must run in
   **one invocation**. A baseline is not reproducible across runs (0.05–0.25 hPa), and not
   even an arm against itself (`notes/nondeterminism_floor_2026_08_20.md`). Array-job
   parallelism is therefore **per storm, never per arm**.

---

## 1. Already on disk: cite it, don't rerun it

| revision item | what exists | where |
|---|---|---|
| Dose-response, all storms (step 3) | g ∈ {0,1,2,3} on 8 storms. g=2 deepens 7/7, cuts deepening error in 6/7 (median 4.46 hPa). Haiyan overshoots. Patricia 30→33% (under-resolved). g=1 is bit-identical to baseline. | `results/skill/suff_convection/`, `notes/result_sufficiency.md` |
| Specificity control that can fail | In-box matched control (87–112% of the group's in-box firing): convection +2.68 vs control +0.02 hPa, 8/8 matched | `results/skill/convection_inboxctl/`, `notes/result_control_audit.md` |
| Unfair control, still loses | Core-matched control over-doses 1.5–9× and costs 0.7–1.6 hPa; treatment wins 6/7; Haishen reversed | `results/skill/conv_corectl4/` |
| Noise floor | Restore-to-normal on min-MSLP: median 0.150, p90 0.369, max 0.608 hPa. Cross-run baseline drift 0.05–0.25 hPa. | `notes/nondeterminism_floor_2026_08_20.md`, `notes/baseline_reproducibility_2026_08_20.md` |
| Non-developer | nondev2013 +0.03 hPa at g=3 | `suff_convection` |
| IBTrACS / HRES (step 10) | 4 storms: GraphCast falls short of best track by 47–88 hPa, HRES by 7–50 hPa. ERA5 is 40–60 hPa too shallow. | `notes/result_hres_gap.md`, `results/hres_compare.npy` |
| Track + spatial readout (step 9) | `MECH_FIELDS=1` stores MSLP and wind grids, the located centre and in-box nodes per arm per lead. **Not yet run on the gain sweep.** | `skill_conv_run.py` |
| IC ensemble (step 15) | `MECH_IC_OFFSETS` implemented | `skill_conv_run.py` |
| Rule-selected storms (step 15) | 80 ERA5-gated storms, 1986–2020, 7 basins: 69 developers, 3 secondary, 8 non-developers. **Gated, not run.** | `flagship_sae/skill_mega_storms.py`, `results/mega_storm_gate*.json` |
| Grid-lock (steps 6, 11) | 6 features stay on the mesh at every rotation angle. Single-feature ablations of all six against coverage+connectivity controls. Rotation score for all 4,096 features. Deleted-norm-matched tiers. | `results/fs_matched_rmse_six.npy`, `fs_rotation_all*.npy`, `fs_rottiers*_rmse.npy`, `notes/result_positional_confirmed.md` |
| Disk radius | Lever saturates by 750 km; the 1500 km disk was conservative | `notes/result_conv_radius_sweep_2026_08_23.md`, `result_conv_r750f_2026_08_23.md` |
| Top-k side effect | Ablating any feature reshuffles ~300 others through top-k competition. Absolute "A moved B" claims need a matched-amplitude control. | `notes/p0_topk_competition_2026_08_20.md` |

**Gap that v1 got right and is still open.** `suff_inboxctl` holds only `baseline` and
`rand-normal`. The gain sweep has **never been run against the in-box control at matching
gains**, so the sufficiency result (step 3) still has no control that could have failed. → J1.

---

## 2. Environment

`CLAUDE.md` says the canonical GraphCast/JAX interpreter is **`.venv-jax/bin/python`** on
the L40S box. Analysis runs on system `python3` there. `~/savar-validation/.venv` is gone,
so `env/README.md` is stale on that point. The `fs_common.py` docstring still names the
authors' venv (`~/graphcast-interpretability/.venv`: jax 0.7.0, `graphcast@sae-hooks` fork,
torch 2.13+cu126). Treat whichever one actually ran the committed verdicts as ground truth.

**Before leaving the L40S box (do this first; it's 2 minutes):**

```bash
.venv-jax/bin/pip freeze > env/requirements-flagship-freeze.txt
.venv-jax/bin/python -c "import jax, graphcast, sys; print(sys.version, jax.__version__, graphcast.__file__)" \
  > env/flagship-provenance.txt
nvidia-smi --query-gpu=name,driver_version --format=csv >> env/flagship-provenance.txt
git add env/requirements-flagship-freeze.txt env/flagship-provenance.txt && git commit -m "Freeze the flagship env"
```

Rebuilding on the cluster:

```bash
python3.11 -m venv .venv-jax && source .venv-jax/bin/activate
pip install -r env/requirements-flagship-freeze.txt     # the fork installs from git, pinned
pip install "jax[cuda12]==<version in the freeze>"       # match the cluster's CUDA 12 driver
python -c "import jax; print(jax.devices())"              # must list a GPU
```

If `graphcast` resolves to PyPI rather than the `sae-hooks` fork, stop. The fork is what
`fs_common`'s interceptor hooks into.

---

## 3. Assets (not in git; `src/weights/` is gitignored)

| file | source | md5 |
|---|---|---|
| `graphcast_flagship_0p25_37lev.npz` | `gs://dm_graphcast/graphcast/params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz` (renamed) | `c3ed5ba70a44691d52a2b3cb6771a88f` |
| `sae_k32_lat4096_lay08.pt` | HuggingFace `theodoremacmillan/sae-graphcast-k32-lat4096-lay08` | `06ba1fddab908b6c78b0e21a324d2f95` |
| `sae_k32_lat4096_lay08.npz` | derived from the `.pt` (W_enc 4096×512, W_dec 512×4096, b_pre 512) | `b0b92a176189a6966c9d70695c06f265` |
| `config.json` | same HF repo | `8f491bc0f8007957608923d105d1b649` |

Simplest is to `rsync src/weights/` from the L40S box, or copy via
`s3://graphcast-rung/` if the cluster can assume the role.

**Also copy** these, which are not weights but feed J1/J5 and are expensive to rebuild:

- `results/inbox_control_skill_conv_storms.json`, `results/inbox_control_skill_sh_storms.json` (in git)
- the i.i.d. dump `~/gc_flagship/fs_iid_dump.npy` + `fs_iid_meta.json`. These are needed only to
  draw *new* in-box controls, for J5's 80 storms. That's 6.7 GB.
- `results/skill/*/ftarget_*.json` (frozen normal references, in git), so `MECH_FTARGET`
  reproduces.

**ERA5** streams from `gs://weatherbench2/datasets/era5/1959-2022-full_37-6h-0p25deg_derived.zarr`
(anonymous). Compute nodes need outbound HTTPS to `storage.googleapis.com`. **If they have
no internet access**, stage the windows on a login node and add an `FS_ERA5_LOCAL` branch to
`fc.load_block`:

- J1–J4, J6, J7 (8 storms, IC−12 h … +96 h, 18 frames): ≈ 0.5 TB.
- J5 (80 storms, −48 … +96 h, 26 frames): ≈ 6–7 TB. Only stage this if J5 is actually going
  to run.

---

## 4. Path portability (one commit, before any job)

```python
# src/fs_common.py
ROOT = pathlib.Path(os.environ.get("CG_ROOT", pathlib.Path(__file__).resolve().parents[1]))
SCRATCH = pathlib.Path(os.environ.get("CG_SCRATCH", "/home/ec2-user/gc_flagship"))
```

Route through `fc.ROOT` only the literal `/home/ec2-user` paths in the scripts these jobs run:
`skill_conv_run.py`, `skill_conv_analyze.py`, `skill_mega_storms.py`, `inbox_control.py`, and
the new baseline script. Leave the other ~115 files alone.

---

## 5. Jobs still to run

Common settings: bf16, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `FS_DEVICE=gpu`, one job per
GPU, every compared arm in one invocation. Storm list from `skill_conv_storms.py` unless noted.
~70 s per 16-step rollout per arm on an L40S; compile ~2–3 min.

### J1. Controlled gain sweep (revision steps 3, 4, 7, 9). ≈ 4 GPU-h. **Do first.**

```bash
MECH_NAME=suff_ctl MECH_GAINS=0,0.5,1,1.25,1.5,2,2.5,3 MECH_FIELDS=1 \
MECH_INBOX_CTL=results/inbox_control_skill_conv_storms.json \
  $PY src/skill_conv_run.py --storm <one storm per array task>
```

- **Code change needed:** today the control gets only `rand-gain-{max(GAINS)}`
  (`skill_conv_run.py:339`). Add a `rand-gain-<g>` arm for every g in `MECH_GAINS`, keeping
  the existing arm's name unchanged so committed verdicts still parse. The in-box control
  must receive **the same gains** in the same invocation. Check that the arm list printed at
  start includes them, then confirm in the verdict.
- This adds 16 arms per storm (8 treatment + 8 control) plus baseline. Check GPU memory with
  a one-storm timing run before launching the array.
- `MECH_FIELDS=1` gives track error and spatial fields for free. Reduce out-of-disk z500,
  t850 and MSLP RMSE on the host and save the scalars, not the full fields.
- **Off-manifold flag:** also log, per gain, the fraction of in-disk node activations above
  each feature's 99.9th percentile in the i.i.d. dump. This is a CPU-side readout.
- Gate: g=0 must reproduce `convection_inboxctl` conv-normal within the p90 floor (0.37 hPa),
  and g=1 must be bit-equivalent to baseline.

### J2. Direction baselines (revision step 8). ≈ 6 GPU-h. New code.

Add `delta_dir(x, fsel, ftarget, nmask, gain, dirs)` to `SAEJax`. It computes the same
per-node `(gain−1)·excess` magnitude as `delta_gain`, but along a supplied direction instead
of each feature's decoder column. Then `‖Δh‖` per node is identical to the treatment by
construction. Arms, all in one invocation per storm:

- **B1:** 5 random unit directions (fixed seeds, orthogonalised against the group's decoder
  span)
- **B3:** difference-of-means direction (raw layer-8 acts, storm-disk nodes minus
  same-location storm-free analogues), from the i.i.d. dump on CPU
- the real decoder directions (`delta_gain`) as the reference arm

Gains {1.5, 2, 3}, 8 storms. Run the same best-gain selection for every arm, or select
leave-one-storm-out. `steer_tc.py` has the earlier same-site random-direction construction
(Mangkhut, ~15% floor) to crib from.

### J3. Pre-patch TC readout (revision step 2, Fig 3). 0 extra GPU-h; fold into J1.

`build_apply_cond` stores `X` **after** `X = X + _d`, so feature 3243 is encoded from the
patched activation. cos(enc₃₂₄₃, dec₂₄₀₁) = 0.179, rank 6 of 4096; the null p99 is 0.119.
Store `X_pre` as a second capture (one extra output in the same graph). Report 3243 from
`X_pre` at step t+1 next to the current post-patch value. Per the P0 result (top-k
reshuffle), also report the in-box control's 3243 under the same gain. An increase only
counts if it beats that.

### J4. Determinism replicate. ≈ 1 GPU-h. Cheap and overdue.

"Nothing in this repo has ever run the same arm twice." Run baseline ×5 and conv-normal ×5
as separate arms in one invocation on Ida and Patricia. This gives a direct per-arm floor,
replacing the inferred one. Required before any 0.1–0.6 hPa number (Patricia, the control
arms) is quoted.

### J5. 80-storm battery (revision step 15). ≈ 60–90 GPU-h. Optional for the camera-ready.

Registry `skill_mega_storms.py` (69 developers + 8 non-developers + 3 secondary). Arms:
baseline, conv-normal, g=2, and the in-box control at both doses. Controls must first be
drawn per storm with `inbox_control.py` (CPU, needs the i.i.d. dump). This is what makes
leave-one-storm-out gain selection, and a non-developer false-alarm rate out of 8, possible.
Time one storm first (`CLAUDE.md`: "time one run before launching many").

### J6. Grid-lock 2954: ablation-type check (revision step 11). ≈ 1 GPU-h.

Zero / mean / resample ablation of 2954 and its matched control on the 6 committed ICs,
+48 h, in one invocation. Everything else in the grid-lock lane is already done (section 1).

### J7. Southern Hemisphere replication. ≈ 2 GPU-h.

`sh_convection` exists for restore-to-normal; add the J1 gain sweep with the SH in-box
controls (`inbox_control_skill_sh_storms.json`). Set `MECH_ALLOW_OVERLAP=1`: it is the
convection group under another name, and forgetting the flag cost a queue cycle once
(commit 179f487).

| job | GPU-h | answers | priority |
|---|---|---|---|
| J1 | ~4 | sufficiency has a control that can fail; off-manifold; track/collateral | 1 |
| J3 | 0 (in J1) | Fig 3 leak | 1 |
| J4 | ~1 | small numbers quotable | 2 |
| J2 | ~6 | SAE direction vs random / diff-of-means | 2 |
| J6 | ~1 | 2954 not a zeroing artefact | 3 |
| J7 | ~2 | hemisphere replication | 3 |
| J5 | 60–90 | rule-selected storms, LOSO, false-alarm rate | 4 (journal) |

---

## 6. Prereg before running (frozen bars)

Write `notes/prereg_cluster_revision.md`, commit it, *then* write any new analysis script
(`CLAUDE.md` rule 1). Minimum content:

- **J1 sufficiency vs control:** per storm, treatment minus in-box control at each gain.
  Claim "specific sufficiency" iff the difference exceeds the J4 floor in ≥ 5 of 7 developing
  storms at g = 2. Patricia is pre-declared under-resolved.
- **Guardrail #9 for J1:** show the control *can* fail. In-box firing must be 87–112% (already
  true); report Haiyan, where the control already costs 2.8 hPa, as the demonstration that
  it is not inert.
- **J2:** report treatment − B1 median and treatment − B3 with bootstrap CIs over storms,
  whichever way they come out. B3 ≥ treatment is reported as a result, not as a failure.
- **J3:** the post-patch to pre-patch drop in 3243 is the leak. The headline uses pre-patch
  only.
- **False alarm:** non-developer deepening > J4 floor at any g ≤ 2 is reported as spurious
  spin-up.

---

## 7. Sanity gates on the new cluster (stop on any failure)

1. `jax.devices()` lists a GPU; md5s in section 3 match.
2. **Zero-patch identity inside one graph:** gain = 1 arm == baseline arm, bit-for-bit.
3. **Cross-hardware reproduction:** Ida conv-normal `d_deepen` = 2.68 hPa (`convection_inboxctl`)
   within the **p90 floor, 0.37 hPa**. Expect bits to differ on a new GPU and XLA autotune;
   only effects ≫ floor are comparable across clusters. Never difference an arm from this
   cluster against one from the L40S box.
4. **Merge gate** (from 070530f): when combining runs, difference an arm that cannot depend
   on what changed (the baseline). Refuse the merge if it moves beyond the floor.

---

## 8. Cluster-specific rules (from `CLAUDE.md`, restated because they bite harder here)

- **No `pgrep` GPU guards.** Use `nvidia-smi --query-compute-apps=used_memory` or the
  scheduler's own GPU allocation. The old `run_skill_conv.sh` guard is exactly the pattern
  that cost ~19 GPU-h.
- **Reserve CPU, not just the card.** WB2 streaming needs ~25 cores per job or it runs 6× slower:
  `-c 24` or more.
- **Crash-resume must not change the arm set.** A resume with a different `MECH_ARMS` silently
  breaks comparability (070530f). On restart, rerun the whole storm.
- **Write job scripts with an editor or Write tool, not heredocs** (the heredoc-ghost failure).

### SLURM wrapper

```bash
#!/bin/bash
#SBATCH -J gc-storm --gres=gpu:1 --constraint="l40s|a100|h100" --mem=128G -c 24 -t 06:00:00
#SBATCH --array=0-7                     # one task per STORM; all arms inside the task
export CG_ROOT=$HOME/causal-graphcast XLA_PYTHON_CLIENT_PREALLOCATE=false FS_DEVICE=gpu
STORMS=(ida2021 michael2018 haishen2020 goni2020 haiyan2013 patricia2015 wilma2005 nondev2013)
$CG_ROOT/.venv-jax/bin/python $CG_ROOT/src/skill_conv_run.py \
  --storm ${STORMS[$SLURM_ARRAY_TASK_ID]}
```

`skill_conv_run.py` loops over all storms internally and has no CLI flags today. Add a
`--storm` filter (or `MECH_STORMS` env var, matching the file's `MECH_*` convention) that
changes nothing else. Per-storm crash-safe outputs (`results/skill/<MECH_NAME>/run_<storm>.npy`) already
exist, so array tasks don't collide.

---

## 9. Bring results home

- Small `verdict.json` / `.npy` to `results/skill/<MECH_NAME>/` (in git); fields and dumps
  to `s3://graphcast-rung/results/`.
- Add `"provenance": {gpu, driver, cuda, jax, cluster}` to every verdict. Cross-cluster
  numbers must be identifiable as such.
- Update the revision spec ledger (step 1) from the verdicts, not from the logs.
