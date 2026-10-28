*Historical document, kept as written; script paths refer to the camera-ready `causal-graphcast` layout (`src/` and `flagship_sae/` are now `graphcast_sae/`, regrouped by experiment; `notes/` is now `docs/notes/` and `docs/prereg/`; `scripts/` launchers are in `scripts/` here).*

# J8 controls drawn from the rebuilt dump (amendment 5): per-storm 80-120% check

Job 24525739. Files: `results/inbox_control_skill_conv_storms_{vort3316,conv2401}.json`.

| storm | f3316 ctl | ratio | | f2401 ctl | ratio | |
|---|---|---|---|---|---|---|
| ida2021 | 3884 | 79.6% | MISS | 3845 | 115% | OK |
| michael2018 | 2533 | 77% | MISS | 2572 | 0/0 | MISS |
| haishen2020 | 2577 | 85% | OK | 2577 | 85% | OK |
| goni2020 | 3696 | 122% | MISS | 3625 | 108% | OK |
| haiyan2013 | 2445 | 94% | OK | 2490 | 115% | OK |
| patricia2015 | 3238 | 113% | OK | 3166 | 95% | OK |
| wilma2005 | 3204 | 107% | OK | 3427 | 0/0 | MISS |
| nondev2013 | 1067 | 85% | OK | 1050 | 112% | OK |

Under amendment 5 as written, a MISS storm has no valid control and counts as not passing.
- f3316: 4 developing storms valid, so the 5/7 bar is UNATTAINABLE. Under the declared rule the f3316
  specific-sufficiency claim cannot be made whatever the sweep shows.
- f2401: 5 valid; passing needs all five.

Design error, mine, recorded here: the 80-120% check was stricter than the draw's own 25% tolerance,
so a control the draw accepts can fail it (Ida 79.6%, Michael 77%, Goni 122%). And for f2401 in
Michael and Wilma the feature never fires in-box climatologically (0.000), so the ratio is 0/0;
the check did not anticipate that. Neither is repaired here. The sweeps still run: treatment
numbers do not depend on the control, and treatment-control is reported for every storm with
the MISS flag.
