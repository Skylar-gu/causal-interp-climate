"""Score results/fs_ida_genesis_v2.npy against the bars in docs/prereg/prereg_ida_genesis_calibrated.md.

Paper: Sec. 3.3; docs/notes/result_ida_genesis_calibrated_2026_08_29.md
Inputs: results/fs_ida_genesis_v2.npy, results/fs_mechanisms_v2.npy (both shipped)
Outputs: results/fs_ida_genesis_v2_verdict.json (shipped)
Run:   # CPU, seconds
    python -m graphcast_sae.storms.ida_genesis_v2_analyze
"""
import json
import numpy as np
from graphcast_sae.paths import RESULTS

d = np.load(RESULTS / "fs_ida_genesis_v2.npy", allow_pickle=True).item()
lab = np.load(RESULTS / "fs_mechanisms_v2.npy", allow_pickle=True).item()
label = np.asarray(lab["label"]).astype(str); z = np.asarray(lab["zscore"]); mech = list(lab["mech"])
base, base2, floor, E, groups, arms = d["base"], d["base2"], d["floor"], d["exposure"], d["groups"], d["arms"]
B = base[-1]

rand = [arms[k]["traj"][-1] - B for k in arms if k.startswith("random_")]
ctl = max(abs(x) for x in rand)
bar = max(3 * ctl, 3 * floor)
print(f"baseline +48h {B:.1f} (repeat {base2[-1]:.1f}, floor {floor:.2f});  random ctl max|Δ| {ctl:.2f}  ->  bar = {bar:.2f} ({100*bar/B:.1f}%)\n")

print(f"{'arm':>18}{'features':>22}{'exposure':>10}{'TC@48h':>9}{'Δ':>8}{'Δ%':>7}   verdict")
out = {}
for name, a in arms.items():
    tr = a["traj"]; dlt = tr[-1] - B; pct = 100 * dlt / B
    fs = a["feats"]; ex = float(np.mean([E[f] for f in fs]))
    passes = abs(dlt) > bar
    if not passes: v = "within bar"
    elif a["coef"] < 0: v = "NECESSARY" if pct <= -15 else ("wrong sign, passes bar" if pct > 0 else "effect < 15%")
    else: v = "SUFFICIENT" if pct >= 15 else ("wrong sign, passes bar" if pct < 0 else "effect < 15%")
    out[name] = dict(feats=fs, coef=a["coef"], tc48=float(tr[-1]), delta=float(dlt), pct=float(pct), exposure=ex, verdict=v, traj=[float(x) for x in tr])
    print(f"{name:>18}{str(fs):>22}{ex:>10.1f}{tr[-1]:>9.1f}{dlt:>+8.1f}{pct:>+7.0f}   {v}")

print("\ngroup identities (calibrated label, z on its own probe, in-box exposure on baseline):")
for g, fs in groups.items():
    print(f"  {g:>14}: " + "  ".join(f"{f}[{label[f]} z={z[f, mech.index(label[f])] if label[f] in mech else float('nan'):+.1f} E={E[f]:.1f}]" for f in fs))

summary = dict(base48=float(B), base2_48=float(base2[-1]), floor=float(floor), ctl=float(ctl), bar=float(bar),
               groups={g: [int(f) for f in fs] for g, fs in groups.items()}, arms=out)
json.dump(summary, open(RESULTS / "fs_ida_genesis_v2_verdict.json", "w"), indent=1)
print("\n-> results/fs_ida_genesis_v2_verdict.json")
