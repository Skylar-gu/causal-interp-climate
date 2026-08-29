"""Ida genesis knockout under CALIBRATED mechanism labels.

From Ida's formation (2021-08-26, cyclone feature ~0) hold a feature GROUP persistently ablated
(coef -1) or dosed (+1) through a 48-h rollout and read feature 3243 summed over the Caribbean+Gulf
box. Moisture / vorticity / shear groups are the three strongest rotation-null-calibrated members of
each mechanism (z > 0 on its own probe, in-box exposure >= half the weakest convection member,
ranked by z); three exposure-matched random controls and a repeat baseline (nondeterminism floor).

Paper: Sec. 3.3 (genesis ingredients on the model's cyclone feature); docs/notes/result_ida_genesis_calibrated_2026_08_29.md
Prereg: docs/prereg/prereg_ida_genesis_calibrated.md (+ amendments 1, 2 inside it)
Inputs: GraphCast params (GRAPHCAST_PARAMS); results/fs_mechanisms_v2.npy (shipped); data/mesh_2to6_geom.npy
Outputs: results/fs_ida_genesis_v2.npy  (scored by ida_genesis_v2_analyze.py -> results/fs_ida_genesis_v2_verdict.json)
Run:   # JAX env, GPU (~46 GB), ~6 min
    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -m graphcast_sae.storms.steer_ida_genesis_v2
"""
import os, sys, time
os.environ["FS_DEVICE"] = "gpu"; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np, jax.numpy as jnp, xarray as xr, dataclasses
import graphcast_sae.common.fs_common as fc
from graphcast import data_utils, rollout

TC = 3243
IC = "2021-08-26"
H = 8                                                          # 48 h
BOX = dict(lat=(10, 33), lon=(-98, -58))                       # Caribbean -> Gulf, Ida's track
CONV = [2401, 2067, 3174]                                      # the committed convection triplet
SEEDS = [7, 8, 9]
OUT = fc.ROOT / "results/fs_ida_genesis_v2.npy"


def numpyify(ds):
    return xr.Dataset({v: (ds[v].dims, np.asarray(ds[v].values)) for v in ds.data_vars},
                      coords={k: ds.coords[k] for k in ds.coords})


def build_io(t0, tc, H):
    blk, times, statics = fc.load_block(np.datetime64(t0), nframes=2 + H)
    w = blk.assign_coords(time=(times - times[0]).astype("timedelta64[ns]"))
    for v in list(w.data_vars):
        if "time" in w[v].dims: w[v] = w[v].expand_dims("batch")
    for v in fc.STATIC_VARS: w[v] = statics[v]
    w = w.assign_coords(datetime=(("batch", "time"), times[None, :].astype("datetime64[ns]")))
    return data_utils.extract_inputs_targets_forcings(
        w, target_lead_times=slice("6h", f"{6*H}h"), **dataclasses.asdict(tc))


def main():
    lab = np.load(fc.ROOT / "results/fs_mechanisms_v2.npy", allow_pickle=True).item()
    label = np.asarray(lab["label"]).astype(str); zscore = np.asarray(lab["zscore"]); mech = list(lab["mech"])
    assert all(label[f] == "ascent" for f in CONV), "convection triplet must calibrate as ascent"
    geom = np.load(fc.MESH_GEOM, allow_pickle=True).item()
    mlat = geom["lat"]; mlon = np.where(geom["lon"] > 180, geom["lon"] - 360, geom["lon"])
    inbox = (mlat >= BOX["lat"][0]) & (mlat <= BOX["lat"][1]) & (mlon >= BOX["lon"][0]) & (mlon <= BOX["lon"][1])

    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply(mc, tc, stats, sae=sae, bf16=True)
    apply = fc.make_apply(params, rf, patched=True)
    noop = fc.noop_patch(sae)

    def box_codes(acts):
        X = jnp.asarray(np.asarray(acts, np.float32).reshape(-1, fc.D_IN))
        return np.asarray(sae.codes(X))[inbox].sum(0)          # (F,) in-box activation sum

    inp, tgt, frc = build_io(IC, tc, H)
    tct = tgt.time.isel(time=slice(0, 1))
    for c in ("datetime",):
        if c in tgt.coords: tgt = tgt.drop_vars(c)
        if c in frc.coords: frc = frc.drop_vars(c)

    def persistent_roll(patch, keep_all=False):
        """apply `patch` at EVERY step; return TC-in-box trajectory (+ (H,F) in-box sums if keep_all)."""
        cur = inp; traj = []; allf = []
        for h in range(H):
            ct = tgt.isel(time=slice(h, h+1)).assign_coords(time=tct); cf = frc.isel(time=slice(h, h+1)).assign_coords(time=tct)
            preds, acts = apply(cur, ct, cf, patch)
            bc = box_codes(acts); traj.append(float(bc[TC]))
            if keep_all: allf.append(bc)
            preds = numpyify(preds)
            if h < H-1: cur = rollout._get_next_inputs(cur, xr.merge([preds, cf])).assign_coords(time=cur.coords["time"])
        return (np.array(traj), np.stack(allf)) if keep_all else np.array(traj)

    t0 = time.time()
    base, boxall = persistent_roll(noop, keep_all=True)
    base2 = persistent_roll(noop)
    E = boxall.max(0)                                          # exposure per feature: max over leads of in-box sum
    floor = abs(base[-1] - base2[-1])
    print(f"baseline TC-feature genesis (box, +6h..+48h): {np.array2string(base, precision=1)}", flush=True)
    print(f"  Ida spins up: {base[0]:.1f} -> {base[-1]:.1f};  repeat baseline +48h {base2[-1]:.1f}  floor={floor:.2f}  "
          f"({(time.time()-t0)/60:.1f}m)\n", flush=True)

    # ---- feature selection: calibrated label x baseline exposure, nothing else ----
    E_MIN = 0.5 * min(E[f] for f in CONV)                      # prereg amendment 1: exposure floor
    def top_by_exposure(name, exclude, n=3):
        """label == name, z > 0 on its own probe (fires where the mechanism is anomalously STRONG, not
        weak), exposure >= half the weakest convection member; rank by z."""
        mi = mech.index(name)
        cand = [f for f in np.where(label == name)[0]
                if zscore[f, mi] > 0 and E[f] >= E_MIN and f not in exclude and f != TC]
        cand = sorted(cand, key=lambda f: -zscore[f, mi])[:n]
        return [int(f) for f in cand]
    print(f"  exposure floor E_MIN = {E_MIN:.1f}", flush=True)
    groups = {"convection": list(CONV)}
    groups["ascent_byrule"] = top_by_exposure("ascent", set(CONV))
    groups["moisture"] = top_by_exposure("q600", set(CONV))
    groups["vorticity"] = top_by_exposure("vort850", set(CONV))
    groups["shear"] = top_by_exposure("shear", set(CONV))
    used = set(f for g in groups.values() for f in g)
    rng_pool_note = {}
    for s in SEEDS:
        rng = np.random.default_rng(s); pick = []
        for f0 in CONV:
            for lo, hi in ((0.5, 2.0), (0.25, 4.0)):
                pool = [f for f in range(len(E)) if lo*E[f0] <= E[f] <= hi*E[f0]
                        and label[f] != "ascent" and f != TC and f not in used and f not in pick]
                if pool: break
            rng_pool_note[(s, f0)] = (lo, hi, len(pool))
            pick.append(int(rng.choice(pool)))
        groups[f"random_s{s}"] = pick
    for g, fs in groups.items():
        assert TC not in fs, f"outcome feature {TC} inside group {g}"
        zs = " ".join(f"{f}[{label[f]},z={zscore[f, mech.index(label[f])] if label[f] in mech else float('nan'):+.1f},E={E[f]:.1f}]" for f in fs)
        print(f"  group {g:>14}: {zs}", flush=True)
    for k, v in rng_pool_note.items():
        if v[0] != 0.5: print(f"  NOTE random seed {k[0]} match for {k[1]} widened to [{v[0]}x,{v[1]}x], pool {v[2]}")
    print(flush=True)

    # ---- arms ----
    arms = {}
    def run(name, feats, coef):
        tr = persistent_roll(fc.coef_patch(sae, feats, coef))
        arms[name] = dict(feats=list(map(int, feats)), coef=coef, traj=tr)
        d = tr[-1] - base[-1]
        print(f"  {name:>26} {'ablate' if coef < 0 else 'dose':>6}  TC@48h {tr[-1]:7.1f}  Δ {d:+7.1f}  ({100*d/base[-1]:+.0f}%)"
              f"   {(time.time()-t0)/60:.1f}m", flush=True)
    for g in ["convection", "ascent_byrule", "moisture", "vorticity", "shear"]:
        if not groups[g]: print(f"  {g:>26}  NO EXPOSURE — no calibrated {g} feature fires in the box"); continue
        run(f"{g}", groups[g], -1.0)
    for g in ["convection", "ascent_byrule", "moisture", "vorticity", "shear"]:
        if groups[g]: run(f"{g}_dose", groups[g], +1.0)
    for s in SEEDS: run(f"random_s{s}", groups[f"random_s{s}"], -1.0)
    allfour = sorted(set(groups["convection"] + groups["moisture"] + groups["vorticity"] + groups["shear"]))
    run("all_four", allfour, -1.0)
    run("conv+moist_dose", sorted(set(groups["convection"] + groups["moisture"])), +1.0)

    np.save(OUT, dict(base=base, base2=base2, floor=floor, exposure=E, groups=groups, arms=arms,
                      box=BOX, ic=IC, H=H, tc=TC, label_file="results/fs_mechanisms_v2.npy"), allow_pickle=True)
    print(f"\n{(time.time()-t0)/60:.1f}m  -> {OUT}")


if __name__ == "__main__":
    main()
