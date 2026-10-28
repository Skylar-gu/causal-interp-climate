*Historical document, kept as written; script paths refer to the camera-ready `causal-graphcast` layout (`src/` and `flagship_sae/` are now `graphcast_sae/`, regrouped by experiment; `notes/` is now `docs/notes/` and `docs/prereg/`; `scripts/` launchers are in `scripts/` here).*

# Amendment 6 to prereg_cluster_revision.md: re-run J8 on the original box with the original dump

Written 2026-10-24, AFTER the J8 results from the rebuilt dump were seen
(`notes/result_j8_2026_10_20.md`: f2401 4/7 fails; f3316 6/7 descriptive, 3/7 under amendment 5).
The original L40S box, which holds the original `fs_iid_dump.npy`, became reachable after that.
This is a replication on the pre-registered instrument, chosen after seeing the first results, and
is reported as such. It supersedes amendment 5 (rebuilt dump, 80-120% check) for the J8 verdict;
the amendment-5 run is kept and reported alongside it, not deleted.

**What runs.** Exactly amendment 4's J8 design (gains, storms, control at every gain, pre-patch
readout, deterministic XLA, f3316 and f2401 separately, seed 7 draw rule, same bans), on the L40S,
with controls drawn from the original 160-window dump. Results go to `vort3316_ctl_orig` and
`conv2401_ctl_orig`, so they never overwrite the H100 runs. Script: `scripts/j8_orig_box.sh`.

**Gates, in order; each one stops the run.**
1. Dump identity: `fs_iid_meta.json` has `n_windows = 160` and seed 0, and its start times equal
   `extract_iid_subset.frozen_starts()`. The sha256 is recorded.
2. Code identity: re-drawing J1's convection control from this dump reproduces
   `results/inbox_control_skill_conv_storms.json` exactly: identical picks, `conv_inbox` within 1e-6.
   This is amendment 4's original gate. If it fails, nothing is drawn.
3. Ida first, both features: `gain-1` == `baseline` bit-for-bit; the baseline deepening error is within
   the J4 floor (0.25 hPa) of J1's (15.10). Bit-equality with J1 is NOT required, because J1 ran on the
   H100. A miss means L40S runs are not comparable to J1: instrument failure, and the other storms are not run.

**Control validity.** Same as amendment 4: the draw's own rule (within 25%, widened once if empty)
and no extra ratio check. Amendment 5's 80-120% check is dropped, because it was stricter than the
draw (a design error recorded in `notes/result_j8_controls_2026_10_20.md`). Dropping it after seeing the
data is part of what makes this amendment post hoc. Matched-ratio per storm is still reported.

**Bar and predictions.** Unchanged from amendment 4: treatment-control > 0.25 hPa at g = 2 in >= 5/7.
Expectation, stated now: treatment curves within about 0.5 hPa of the H100 run; f2401 still fails
(its failures, Haishen, Haiyan and Patricia, come from the treatment arm, not the control).
