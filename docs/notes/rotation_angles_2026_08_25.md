*Historical document, kept as written; script paths refer to the camera-ready `causal-graphcast` layout (`src/` and `flagship_sae/` are now `graphcast_sae/`, regrouped by experiment; `notes/` is now `docs/notes/` and `docs/prereg/`; `scripts/` launchers are in `scripts/` here).*

# Grid-locked = stays under rotation at every angle: five-angle check (2026-08-25)

Three more all-feature rotation runs (`src/rotation_all.py`, `RA_ROLL` = 180/540/1080
cells = 45/135/270 deg; ICs 2020-01-10, 2020-07-10) join the 180 and 90 deg runs. Files:
`results/fs_rotation_all_{45,135,270}.npy`; summary `src/rotation_angles_summary.py`;
launcher/log `scripts/rot_angles.sh`, `out/rot_angles.log`. Each run: ~3,940 live features,
~3,920 with power; 9-11 features with positional score >= 0.5 per angle.

**Grid-locked at EVERY angle (powered everywhere, stay - follow >= 0.5): six features —
f2954, f2075, f407, f2585, f2989, f3535.** f1331 stays at every angle (0.84-0.87) but its
footprint coincides with the rotated pattern at 135 deg (follow 0.40), so it misses the cut on
the difference score at that one angle.

| feature | 180 | 90 | 45 | 135 | 270 | min positional |
|---|---|---|---|---|---|---|
| f2954 | .95/.01 | .95/.03 | .95/.12 | .95/.00 | .95/.03 | **+0.84** |
| f2075 | .85/-.04 | .86/.23 | .86/.24 | .85/.30 | .83/.18 | **+0.55** |
| f656 | .64/.01 | .48/.03 | .52/.12 | .57/.00 | .44/.04 | +0.40 (borderline) |
| f3680 | .40/.01 | .20/.02 | .27/.02 | .28/.00 | .20/.03 | +0.18 (weakly positional) |
| f560 | .47/.05 | .07/.02 | .20/.03 | .20/.00 | .13/.02 | +0.05 (weakly positional) |
| f3404 (land-shaped ctrl) | .01/.54 | .02/.01 | .02/.02 | .01/.02 | .01/.01 | content (follows at 180, silent otherwise) |
| f2474 (land mask) | -.02/.40 | .22/-.06 | .15/-.03 | .02/.02 | .11/-.08 | content |
| f2109 (convection) | .00/.01 | .00/.00 | .00/.00 | .02/.02 | .00/.00 | content (does not fire on the rotated planet) |

(cells are stay/follow.)

**Rank agreement.** Over all powered features Spearman between angles is ~0 (-0.36..+0.70):
noise around zero for ~3,900 inert features. Among features scoring >= 0.3 at either angle
(n = 16-18) it is +0.64..+0.92 at every pair. The 2026-08-20 note's "ranking does not replicate
at 90 deg" was the all-feature number; the top of the ranking replicates at every angle.

**Consequence for the paper.** "Grid-locked" is defined by staying under rotation at all five
angles. f2954 (+1.50 m) and f2075 (+0.06 m) are grid-locked and in the single-feature battery;
f3680 (+1.22 m) and f560 (+0.74 m) are load-bearing but only weakly positional, and are no
longer called grid-locked; f656 is borderline. The rot3/rot6 tier ablations
(`fs_rottiers_rmse.npy`: rot3 = f2954, f2075, f1331; rot6 = + f2989, f407, f2585) cover the
robust set (+2.7 m / +4.1 m, day-5 z500, 8 ICs). f407/f2585/f2989/f3535 have no single-feature
ablation yet.
