*Historical document, kept as written; script paths refer to the camera-ready `causal-graphcast` layout (`src/` and `flagship_sae/` are now `graphcast_sae/`, regrouped by experiment; `notes/` is now `docs/notes/` and `docs/prereg/`; `scripts/` launchers are in `scripts/` here).*

# J8 dump-equivalence gate: FAILED (instrument failure, recorded as such)

Job 23576867 (H100, deterministic ops), `scripts/rev_j8_controls.sbatch`, prereg amendment 4.

The 12 i.i.d. windows `inbox_control.py` reads were rebuilt on Rorqual (`src/extract_iid_subset.py`,
`$CG_SCRATCH/fs_iid12_dump.npy`) and J1's convection control re-drawn from them.

**Gate: per-storm in-box convection activation within 2% of the committed file. Worst: 3.44%. FAILED.**
Per the prereg, the J8 controls were not drawn from this dump.

| storm | 2401 | 2067 | 3174 |
|---|---|---|---|
| ida2021 | +0.8% | +3.0% | +2.5% |
| michael2018 | 0 | 0 | -1.2% |
| haishen2020 | -0.3% | -2.6% | 0.0% |
| goni2020 | -0.2% | **+3.4%** | +1.6% |
| haiyan2013 | -2.0% | -0.2% | +0.9% |
| patricia2015 | -0.6% | -0.2% | -0.1% |
| wilma2005 | 0 | -3.2% | -0.1% |
| nondev2013 | +0.6% | -0.5% | -1.0% |

5 of 24 values exceed 2%; signs are mixed (no systematic offset). Re-drawn control features match
J1's in 1 of 8 storms exactly and share at least one feature in 6 of 8; the re-drawn controls still
land at 80-113% of the group's in-box firing (median 97%), the same matching quality as J1's.

Reading: the differences look like L40S-vs-H100 bf16 noise rather than a wrong sample (same dates,
same mean/std scale, no bias), but the 2% bar was set before looking and it was missed. That bar was
also never calibrated against a known hardware-only difference, so it may have been too tight; that is
a statement about the bar, not a reason to pass it.

Artefact kept: `results/inbox_control_skill_conv_storms_conv_check.json`.
