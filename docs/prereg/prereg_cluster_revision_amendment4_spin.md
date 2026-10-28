*Historical document, kept as written; script paths refer to the camera-ready `causal-graphcast` layout (`src/` and `flagship_sae/` are now `graphcast_sae/`, regrouped by experiment; `notes/` is now `docs/notes/` and `docs/prereg/`; `scripts/` launchers are in `scripts/` here).*

# Amendment 4 to prereg_cluster_revision.md: J8 (single-feature gain sweeps, f2401 and f3316)

Frozen 2026-10-19, before any J8 number exists. Revised the same day, still before any J8 number,
from the spin GROUP (2089, 2514, 3316) to the two single features the paper's text reports; the
earlier version is in git (3c8137d). The only related numbers already seen are the August
`gain_conv` (convection group) and `gain_3316` (f3316) runs on Haishen/Ida/Patricia, L40S, random
control at x3 only, rescored on 2026-10-19 with deepening error, and J1 (convection group, 7 storms).
They motivated this run and are not part of its bar.

**Question.** Does amplifying a single convection feature (f2401) or a single vorticity feature
(f3316) improve 96-h deepening error beyond a matched in-box control, under J1's design?

**Design.** Identical to J1 (`scripts/rev_j1.sbatch`) except for the treatment feature and its
control: same 7 developing storms + `nondev2013`, gains {0, 0.5, 1, 1.25, 1.5, 2, 2.5, 3},
control at every gain (`MECH_CTL_ALL_GAINS=1`), pre-patch readout (`MECH_PRE=1`), `MECH_FIELDS=1`,
deterministic XLA, H100, live analogues for the normal reference. One task per storm.

- **J8v:** f3316 alone, `MECH_NAME=vort3316_ctl`.
- **J8c:** f2401 alone, `MECH_NAME=conv2401_ctl` (`MECH_ALLOW_OVERLAP=1`: 2401 is a member of the
  convection group by design).

**Control.** One feature per storm, drawn by `src/inbox_control.py` with J1's rule unchanged
(in-box climatological firing from `fs_iid_dump.npy`, 12 windows, within 25%, seed 7), with the
treatment set by `MECH_FEATS`. Banned from both draws: the convection group (2401, 2067, 3174), the
spin group (2089, 2514, 3316), TC feature 3243, global-rate RANDOM_CTRL. Written to
`results/inbox_control_skill_conv_storms_vort3316.json` and `..._conv2401.json` and committed before
any arm runs. The draw script first re-draws J1's convection-group control and must reproduce the
committed file exactly.

**Deviation, declared before running: the dump is rebuilt.** `fs_iid_dump.npy` was never copied to
Rorqual and the L40S copy is not available. `src/extract_iid_subset.py` regenerates only the 12 of
its 160 frozen windows that `inbox_control.py` reads (same seed-0 sample, same forward pass and
layer-8 capture), on the H100 with deterministic ops instead of the L40S. Gate: re-drawing J1's
convection control from the rebuilt dump must reproduce the committed per-storm in-box convection
activation (`conv_inbox`) within 2% for every feature in every storm. If not, the rebuilt dump is
not equivalent, that is reported as an instrument failure, and the J8 controls are not drawn from it.
Whether the re-drawn control FEATURES match the committed ones is reported but is not the gate: a
bf16-level difference can move a candidate across the 25% tolerance edge and change a seeded draw
without the dump being meaningfully different.

**Bar, each feature separately, copied from J1.** Specific sufficiency is claimed for a feature iff
(treatment - in-box control) deepening-error reduction at g = 2 exceeds the J4 floor (0.25 hPa) in
>= 5 of 7 developing storms. Patricia counts like any storm.
- Guardrail #9: if no storm shows a control effect beyond the floor at any gain, that feature's
  control is reported as possibly inert and its claim is not made.
- Gates: `gain-1` == `baseline` bit-for-bit; each storm's `baseline` must match J1's `suff_ctl`
  baseline within the floor, otherwise the run is an instrument failure.
- Every gain is reported; g = 2 is fixed in advance here.

**Predictions, written before running.**
- f3316: monotone improvement through g = 3 with no overshoot inside the sweep (August curve); passes.
- f2401: follows the convection group's J1 pattern, smaller in size (2401 carries 88% of the group's
  restore-to-normal effect); passes at g = 2 with Haishen and Haiyan overshooting.
- If a feature fails, the paper's amplification claim for it is dropped, not softened.

**Order.** Ida alone for both features first (timing probe and gates), then the remaining tasks.
