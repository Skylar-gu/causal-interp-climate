"""LV-1..LV-4 — verify concept labels by INTERVENTION, from data already on disk.

WHY. Every concept group here was labelled by SPATIAL CO-OCCURRENCE:
    z[f,m] = mean over feature f's ACTIVE nodes of the standardized mechanism field m
i.e. "this feature fires where shear is high". That is not evidence the feature ENCODES
shear, and it has produced one proven failure and two suspected ones:

  * feature 3243 was labelled `atm_river`. It is the TC FEATURE. Ablating it removes 122% of
    Hurricane Ida's deepening. Nothing in the labelling pipeline could catch this.
  * all four `t850` features score HIGHEST on orography (+4.9) and NEGATIVE on t850 (-3.6..-3.1)
  * all four `z500`, and two of four `vort850`, were chosen on NEGATIVE z -- they fire where
    the field is anomalously LOW.

THE TEST. Dose a concept and ask whether the physical field it is NAMED for responds more
than every other field. That is the question the label claims to answer, posed as an
intervention. No GPU needed: respop.py already dosed all 10 concepts at 4 ICs and stored
RMS[ic, arm, lead, field] for all 9 fields. This rescores it for SPECIFICITY instead of
detection.

BARS, frozen here before the numbers are read
  LV-1 VERIFIED     the named field is the top responder by SNR
  LV-2 WRONG-FIELD  a different field responds more -> the label names the wrong mechanism
  LV-4 INERT        no field clears SNR 3 -> the dose does not move the atmosphere at all,
                    so the label is UNTESTED by this instrument, not refuted
Floor = the nf0/nf1 unpatched pair, per field per lead. Established failable in the respop
run: non-zero at every lead, and nf1 scored as an arm reaches only SNR 1.01 against a bar of 3.

A concept can be correctly labelled and still be INERT here, and a concept that is INERT is
not thereby mislabelled. Both are reported; neither is read as the other.

Paper: Sec. 3, mechanism labels that define the intervention groups
Inputs: results/fs_atlas_extra.npy (not shipped, see docs/REPRODUCE.md); results/fs_respop.npy (not shipped, see docs/REPRODUCE.md)
Outputs: results/fs_label_verify.json
Run:   # JAX env, CPU
    python -m graphcast_sae.atlas.label_verify_score
"""
import json
from pathlib import Path

import numpy as np

from graphcast_sae.paths import REPO_ROOT as ROOT
D = np.load(ROOT / "results/fs_respop.npy", allow_pickle=True).item()

RMS, arms, fields, own = D["RMS"], list(D["arms"]), list(D["fields"]), D["own"]
concepts = list(D["names"])
i_nf0, i_nf1 = arms.index("nf0"), arms.index("nf1")

# floor per (lead, field) = the MEASURED numeric floor, defined exactly as respop_score.py
# does it: the mean response magnitude of the two UNPATCHED rolls. Verified failable there --
# non-zero at every lead, and nf1 scored as an arm reaches only SNR 1.01 against a bar of 3.
floor = np.maximum(RMS[:, [i_nf0, i_nf1]].mean((0, 1)), 1e-30)   # (lead, field)

print("=" * 92)
print("LABEL VERIFICATION BY INTERVENTION — does the named field actually respond most?")
print("=" * 92)
print(f"{len(concepts)} concepts x {len(fields)} fields, {RMS.shape[0]} ICs, dose gamma="
      f"{D['gamma']}, {D['S']} steps\n")
print(f"  {'concept':<15}{'named field':<12}{'SNR named':>10}{'top field':>11}"
      f"{'SNR top':>9}   verdict")

rows, ver = {}, {}
for c in concepts:
    ia = arms.index(c)
    # SNR per field = response above the unpatched floor, at the lead of max response
    snr = RMS[:, ia].mean(0) / floor                            # (lead, field)
    per_field = snr.max(0)                                      # best lead per field
    nf = own[c]
    jn = [k for k, f in enumerate(fields) if f.lower() == nf.lower()][0]
    nf = fields[jn]                                             # normalise case
    jt = int(np.argmax(per_field))
    sn, st = float(per_field[jn]), float(per_field[jt])
    if st < 3.0:
        v = "LV-4 INERT (dose moves nothing; label UNTESTED)"
    elif jt != jn:
        v = f"LV-2 WRONG-FIELD -> responds as {fields[jt]}"
    else:
        v = "LV-1 VERIFIED"
    rows[c] = dict(named=nf, snr_named=sn, top=fields[jt], snr_top=st, verdict=v,
                   per_field={f: float(per_field[k]) for k, f in enumerate(fields)})
    ver[c] = v.split()[0]
    print(f"  {c:<15}{nf:<12}{sn:>10.1f}{fields[jt]:>11}{st:>9.1f}   {v}")

print("\n" + "-" * 92)
n1 = sum(1 for v in ver.values() if v == "LV-1")
n2 = sum(1 for v in ver.values() if v == "LV-2")
n4 = sum(1 for v in ver.values() if v == "LV-4")
print(f"  VERIFIED {n1}/{len(concepts)}   WRONG-FIELD {n2}   INERT (untested) {n4}")
print("-" * 92)

print("\nFull concept x field SNR matrix (max over lead; named field marked *):")
print(f"  {'concept':<15}" + "".join(f"{f:>9}" for f in fields))
for c in concepts:
    pf = rows[c]["per_field"]
    print(f"  {c:<15}" + "".join(
        (f"{pf[f]:>8.1f}" + ("*" if f == rows[c]["named"] else " ")) for f in fields))

# cross-check against the co-occurrence labelling that built these groups
print("\nCo-occurrence label evidence for the same groups (z on own mechanism, 11-col atlas):")
a = np.load(ROOT / "results/fs_atlas_extra.npy", allow_pickle=True).item()
Z = np.asarray(a["z"], float)
N = ["vort850", "q600", "ascent", "shear", "t850", "z500", "jet250", "div250",
     "abslat", "land_sea", "orography"]
G = json.load(open("/tmp/mech_groups.json"))
for c in concepts:
    if c not in N:
        print(f"  {c:<15} not in the 11-column atlas -- no co-occurrence check exists")
        continue
    mi = N.index(c)
    zs = [Z[f][mi] for f in G[c]]
    tops = [N[int(np.argmax(np.abs(Z[f])))] for f in G[c]]
    mism = sum(1 for t in tops if t != c)
    print(f"  {c:<15} own z {np.round(zs,2).tolist()}"
          f"   {mism}/4 score highest on another field"
          f"{'  <-- ' + max(set(tops), key=tops.count) if mism else ''}")

json.dump({c: {k: v for k, v in r.items()} for c, r in rows.items()},
          open(ROOT / "results/fs_label_verify.json", "w"), indent=1)
print("\n-> results/fs_label_verify.json")
