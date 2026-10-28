"""Off-distribution check for the J1/J7 gain sweeps: do the steered convection codes leave the
range the SAE sees in ordinary weather?

skill_conv_run.py (MECH_FIELDS=1) stores, per arm and lead, the maximum of each STEERED
convection code over the intervention disk (rev_analyze.py: `disk_max`, max over leads). This
compares that maximum with the feature's maximum over the 160-window i.i.d. catalog
(`smax` in results/fs_feature_magnitude_stats.npy, 160 x 40,962 node samples).

The pre-registered comparison was against the i.i.d. 99.9th percentile. That bar is vacuous here:
the convection features fire at 0.1-0.4% of nodes, so their i.i.d. q99.9 is far below what any
storm reaches unsteered, and every arm (baseline included) would be flagged. The catalog maximum
is the stricter, informative reference; a ratio > 1 means beyond anything in 160 windows.

Paper: Sec. 3.3 (amplification at alpha = 2 is partly beyond the ordinary range)
Inputs: results/skill/{suff_ctl,sh_suff_ctl}/rev_verdict.json, results/fs_feature_magnitude_stats.npy (shipped)
Outputs: results/offmanifold_check.json
Run:   # any env with numpy, CPU
    python -m graphcast_sae.storms.offmanifold_check
"""
import json

import numpy as np

from graphcast_sae.paths import RESULTS

CONV = [2401, 2067, 3174]           # order of disk_max in rev_verdict.json (skill_conv_storms.CONV)
GAINS = ["0", "0.5", "1", "1.25", "1.5", "2", "2.5", "3"]

smax = np.load(RESULTS / "fs_feature_magnitude_stats.npy", allow_pickle=True).item()["smax"]
ref = {f: float(smax[f]) for f in CONV}
out = {"reference": "smax over the 160-window i.i.d. catalog", "smax": ref}
for run in ["suff_ctl", "sh_suff_ctl"]:
    v = json.load(open(RESULTS / f"skill/{run}/rev_verdict.json"))
    ratio = {}
    for storm, m in v["metrics"].items():
        if m.get("nondev"):
            continue
        ratio[storm] = {g: [round(x / ref[f], 3) for x, f in zip(m["arms"][f"gain-{g}"]["disk_max"], CONV)]
                        for g in GAINS}
    count = {g: sum(r[g][0] > 1 for r in ratio.values()) for g in GAINS}
    out[run] = dict(ratio_to_iid_max=ratio, n_storms=len(ratio), n_2401_beyond_iid_max=count)
    print(f"{run}: storms where f2401 exceeds its i.i.d. max, by gain:", count)
    print("   ratio at gain 2 (f2401):", {s: r["2"][0] for s, r in ratio.items()})
json.dump(out, open(RESULTS / "offmanifold_check.json", "w"), indent=1)
print("->", RESULTS / "offmanifold_check.json")
