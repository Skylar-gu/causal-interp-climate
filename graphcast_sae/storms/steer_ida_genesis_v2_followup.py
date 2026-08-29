"""Follow-up to steer_ida_genesis_v2 (prereg amendment 2): single-feature decomposition of the
vorticity and convection groups, plus the original (39c8e9b) vorticity group 3861/2514/2089, same protocol.

Paper: Sec. 3.3; docs/notes/result_ida_genesis_calibrated_2026_08_29.md (3316 alone carries 83% of the spin effect)
Inputs: GraphCast params (GRAPHCAST_PARAMS); results/fs_ida_genesis_v2.npy (the groups)
Outputs: results/fs_ida_genesis_v2_followup.npy
Run:   # JAX env, GPU (~46 GB), ~3 min
    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -m graphcast_sae.storms.steer_ida_genesis_v2_followup
"""
import os, sys, time
os.environ["FS_DEVICE"] = "gpu"; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np, jax.numpy as jnp, xarray as xr, dataclasses
import graphcast_sae.common.fs_common as fc
from graphcast import data_utils, rollout
from graphcast_sae.storms.steer_ida_genesis_v2 import TC, IC, H, BOX, numpyify, build_io

OLD_VORT = [3861, 2514, 2089]
OUT = fc.ROOT / "results/fs_ida_genesis_v2_followup.npy"


def main():
    main_res = np.load(fc.ROOT / "results/fs_ida_genesis_v2.npy", allow_pickle=True).item()
    groups = main_res["groups"]; base_main = main_res["base"]
    geom = np.load(fc.MESH_GEOM, allow_pickle=True).item()
    mlat = geom["lat"]; mlon = np.where(geom["lon"] > 180, geom["lon"] - 360, geom["lon"])
    inbox = (mlat >= BOX["lat"][0]) & (mlat <= BOX["lat"][1]) & (mlon >= BOX["lon"][0]) & (mlon <= BOX["lon"][1])

    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply(mc, tc, stats, sae=sae, bf16=True)
    apply = fc.make_apply(params, rf, patched=True)
    noop = fc.noop_patch(sae)

    def tc_in_box(acts):
        X = jnp.asarray(np.asarray(acts, np.float32).reshape(-1, fc.D_IN))
        return float(np.asarray(sae.codes(X))[inbox, TC].sum())

    inp, tgt, frc = build_io(IC, tc, H)
    tct = tgt.time.isel(time=slice(0, 1))
    for c in ("datetime",):
        if c in tgt.coords: tgt = tgt.drop_vars(c)
        if c in frc.coords: frc = frc.drop_vars(c)

    def persistent_roll(patch):
        cur = inp; traj = []
        for h in range(H):
            ct = tgt.isel(time=slice(h, h+1)).assign_coords(time=tct); cf = frc.isel(time=slice(h, h+1)).assign_coords(time=tct)
            preds, acts = apply(cur, ct, cf, patch)
            traj.append(tc_in_box(acts))
            preds = numpyify(preds)
            if h < H-1: cur = rollout._get_next_inputs(cur, xr.merge([preds, cf])).assign_coords(time=cur.coords["time"])
        return np.array(traj)

    t0 = time.time()
    base = persistent_roll(noop)
    print(f"baseline +48h {base[-1]:.1f} (main run {base_main[-1]:.1f})", flush=True)
    arms = {}
    def run(name, feats, coef=-1.0):
        assert TC not in feats
        tr = persistent_roll(fc.coef_patch(sae, feats, coef)); arms[name] = dict(feats=list(map(int, feats)), coef=coef, traj=tr)
        d = tr[-1] - base[-1]
        print(f"  {name:>22} {feats!s:>20}  TC@48h {tr[-1]:7.1f}  Δ {d:+7.1f}  ({100*d/base[-1]:+.0f}%)   {(time.time()-t0)/60:.1f}m", flush=True)
    for f in groups["vorticity"]: run(f"vort_single_{f}", [f])
    run("vort_old_group", OLD_VORT)
    for f in groups["convection"]: run(f"conv_single_{f}", [f])
    np.save(OUT, dict(base=base, arms=arms, groups=groups, old_vort=OLD_VORT), allow_pickle=True)
    print(f"\n{(time.time()-t0)/60:.1f}m  -> {OUT}")


if __name__ == "__main__":
    main()
