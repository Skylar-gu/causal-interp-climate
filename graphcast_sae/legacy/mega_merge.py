"""Merge the mega_sweep batches into one results/mega_storm_gate.json.

SELECTION RULE, fixed here BEFORE the merged numbers were looked at, so the battery cannot
be tuned after the fact:

  1. pool     = every accepted candidate from every batch file, developers and non-developers
  2. dedupe   = drop anything within 7 days and 1500 km of another candidate (batches cover
                disjoint years, but a Southern-Hemisphere season straddles the year boundary)
  3. analogs  = a storm needs >= 3 ERA5-quiet analogs offered, or it is rejected. ida2021
                shipped with 1 surviving analog and the data-gate rule flags that as below
                standard; a storm that cannot even be OFFERED 3 does not enter.
  4. spread   = at most ONE storm per (basin, year). Offsets do not buy independence and
                neither do two storms from the same season in the same basin.
  5. balance  = round-robin across basins, strongest ERA5 deepening first within each basin,
                until TARGET is reached. A balanced design, not a climatological sample:
                the question is whether the convection handle holds ACROSS basins.

Everything not selected is kept in `rejected` with its reason, so the file is the full
evidence for accepted AND rejected candidates.

Paper: not in the paper; kept for provenance only
Inputs: none beyond the arguments above
Outputs: results/mega_storm_gate.json (merged from the per-batch mega_storm_gate_<tag>.json files)
Run:   # JAX env, CPU
    python -m graphcast_sae.legacy.mega_merge
"""
import argparse
import glob
import json
import os
import time

import numpy as np

from graphcast_sae.legacy.mega_sweep import OUT, clash_with_existing, km
from graphcast_sae.paths import MESH_GEOM

TARGET = 72
TARGET_NONDEV = 8
MIN_ANALOGS = 3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--no-pin", action="store_true")
    a = ap.parse_args()
    target = a.target
    # PINNING. Batches are delivered incrementally and the GPU starts on the first one, so
    # a later merge must never rename, re-box or drop a storm already handed over. Every
    # name already in results/mega_storm_gate.json is pinned into the selection; the
    # round-robin then fills the remaining slots. Without this, adding batch 2 would
    # re-rank every basin and silently invalidate runs already on disk.
    pinned = set()
    if os.path.exists(OUT) and not a.no_pin:
        pinned = {c["name"] for c in json.load(open(OUT)).get("accepted", [])}
        print(f"pinning {len(pinned)} storms already delivered")
    files = sorted(glob.glob(OUT.replace(".json", "_b*.json")))
    pool, rej = [], []
    for f in files:
        g = json.load(open(f))
        for c in g["accepted"]:
            c["batch"] = os.path.basename(f)
            pool.append(c)
        rej.extend(g.get("rejected", []))
    print(f"{len(files)} batch files, {len(pool)} accepted candidates, {len(rej)} rejected")

    # RE-COUNT box_nodes on the ROUNDED box edges that will actually ship, with exactly the
    # comparison skill_conv_run.py uses. The sweep counted before rounding, so its number is
    # not guaranteed to be the one the shipped box reproduces; this is the authoritative one.
    geom = np.load(MESH_GEOM, allow_pickle=True).item()
    mlat = np.asarray(geom["lat"], float)
    mlon = np.asarray(geom["lon"], float)
    mlon = np.where(mlon > 180, mlon - 360, mlon)
    for c in list(pool):
        b = c["box"]
        n = int(((mlat >= b["lat"][0]) & (mlat <= b["lat"][1]) &
                 (mlon >= b["lon"][0]) & (mlon <= b["lon"][1])).sum())
        c["box_nodes_sweep"] = c["box_nodes"]
        c["box_nodes"] = n
        if n < 120:
            c["reject"] = f"box holds only {n} mesh nodes, below the 120 floor"
            pool.remove(c)
            rej.append(c)

    keep = []
    for c in sorted(pool, key=lambda r: -r["era5_deepen"]):
        nm = clash_with_existing(c)
        if nm:
            c["reject"] = f"same storm-week as the existing registry entry {nm}"
            rej.append(c)
            continue
        t = np.datetime64(c["ic"])
        dup = next((o for o in keep
                    if abs((t - np.datetime64(o["ic"])) / np.timedelta64(1, "D")) <= 7
                    and km(c["center"][0], c["center"][1], o["center"][0], o["center"][1]) < 1500),
                   None)
        if dup is not None:
            c["reject"] = f"duplicate storm-week of {dup['name']}"
            rej.append(c)
            continue
        if len(c.get("analogs", [])) < MIN_ANALOGS:
            c["reject"] = (f"only {len(c.get('analogs', []))} ERA5-quiet analogs offered, "
                           f"below the {MIN_ANALOGS} floor")
            rej.append(c)
            continue
        keep.append(c)

    dev = [c for c in keep if not c.get("nondev")]
    nd = [c for c in keep if c.get("nondev")]

    # one per basin-year (pinned storms occupy their slot before anything else)
    per, spread = {}, []
    for c in sorted(dev, key=lambda r: (r["name"] not in pinned, -r["era5_deepen"])):
        k = (c["basin"], c["ic"][:4])
        if per.get(k):
            c["reject"] = f"a stronger storm from {k[0]} {k[1]} is already in the battery"
            rej.append(c)
            continue
        per[k] = 1
        spread.append(c)

    # round-robin across basins, strongest first within each
    by = {}
    for c in spread:
        by.setdefault(c["basin"], []).append(c)
    for v in by.values():
        v.sort(key=lambda r: (r["name"] not in pinned, -r["era5_deepen"]))
    sel, i = [], 0
    while len(sel) < target and any(len(v) > i for v in by.values()):
        for b in sorted(by):
            if len(by[b]) > i and len(sel) < target:
                sel.append(by[b][i])
        i += 1
    chosen = {id(c) for c in sel}
    for b in by.values():       # a pinned storm can never be crowded out by the target
        for c in b:
            if c["name"] in pinned and id(c) not in chosen:
                sel.append(c)
                chosen.add(id(c))
    for c in spread:
        if id(c) not in chosen:
            c["reject"] = f"beyond the {target}-storm target under the round-robin rule"
            rej.append(c)

    ndper, ndsel = {}, []
    for c in sorted(nd, key=lambda r: r["era5_deepen"]):
        if len(ndsel) >= TARGET_NONDEV or ndper.get(c["basin"], 0) >= 2:
            c["reject"] = "beyond the non-developer quota"
            rej.append(c)
            continue
        ndper[c["basin"]] = ndper.get(c["basin"], 0) + 1
        ndsel.append(c)

    acc = sorted(sel + ndsel, key=lambda r: (r["basin"], r["ic"]))
    json.dump(dict(generated=time.strftime("%Y-%m-%d %H:%M"), source_files=files,
                   target=target, accepted=acc, rejected=rej), open(OUT, "w"), indent=1)

    d = [c["era5_deepen"] for c in sel]
    print(f"\n{len(sel)} developers + {len(ndsel)} non-developers -> {OUT}")
    print(f"ERA5 deepening {min(d):.1f} - {max(d):.1f} hPa, median {np.median(d):.1f}")
    print(f"secondary (12 - 18.7 hPa): {sum(1 for c in sel if c.get('secondary'))}")
    print(f"box_nodes {min(c['box_nodes'] for c in acc)} - {max(c['box_nodes'] for c in acc)}")
    print(f"analogs offered: min {min(len(c['analogs']) for c in acc)}, "
          f"median {np.median([len(c['analogs']) for c in acc]):.0f}")
    from collections import Counter
    print("by basin:", dict(Counter(c["basin"] for c in acc)))
    print("by decade:", dict(Counter(c["ic"][:3] + "0s" for c in acc)))
    print(f"rejected with a recorded reason: {len(rej)}")

if __name__ == "__main__":
    main()
