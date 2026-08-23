"""Causally-modular decomposition test (a functional alternative to SPD basis recovery).

SPD tried to recover mechanisms structurally (which weight direction is mode m) and hit the
identification wall. This asks the FUNCTIONAL question instead: does the top feature of each
mechanism, when dosed, move a DISTINCT set of output variables? A causally-modular decomposition
gives a near-diagonal mechanism x output signature (convection->precip/ascent, shear->wind,
jet->flow), which is the success SPD couldn't get by basis recovery.

Paper: not in the paper; kept for provenance only
Inputs: results/fs_atlas.npy (not shipped, see docs/REPRODUCE.md); GraphCast params (GRAPHCAST_PARAMS)
Outputs: results/fs_modularity.npy
Run:   # JAX env, GPU (~46 GB)
    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -m graphcast_sae.legacy.steer_modularity
"""
import os, sys, time
os.environ["FS_DEVICE"] = "gpu"; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np, jax.numpy as jnp
import graphcast_sae.common.fs_common as fc

ICS = ["2020-01-05", "2020-07-06", "2021-08-28"]
OUTVARS = [("total_precipitation_6hr", None), ("10m_wind", None), ("2m_temperature", None),
           ("t850", 850), ("z500", 500), ("ascent500", 500)]

def out_fields(preds):
    def g(name, lev=None):
        if name == "10m_wind":
            u = preds["10m_u_component_of_wind"].isel(batch=0, time=0).values
            v = preds["10m_v_component_of_wind"].isel(batch=0, time=0).values
            return np.hypot(np.asarray(u), np.asarray(v))
        if name == "t850":
            return np.asarray(preds["temperature"].isel(batch=0, time=0).sel(level=850).values)
        if name == "z500":
            return np.asarray(preds["geopotential"].isel(batch=0, time=0).sel(level=500).values) / 9.81
        if name == "ascent500":
            return -np.asarray(preds["vertical_velocity"].isel(batch=0, time=0).sel(level=500).values)
        return np.asarray(preds[name].isel(batch=0, time=0).values)
    return {nm: g(nm, lev) for nm, lev in OUTVARS}

def main():
    a = np.load(fc.ROOT / "results/fs_atlas.npy", allow_pickle=True).item()
    z, phys = a["z"], a["phys"]; alive = a["zcnt"] > 300
    mechs = ["vort850", "q600", "ascent", "shear", "jet250"]
    feat = {}
    for m in mechs:
        j = a["node_refs"].index(m); good = np.where(alive)[0]
        feat[m] = int(good[np.argmax(z[good, j])])
    print("mechanism -> top feature:", feat, flush=True)

    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply(mc, tc, stats, sae=sae, bf16=True)
    apply = fc.make_apply(params, rf, patched=True)
    noop = fc.noop_patch(sae)
    lat = np.asarray(fc.load_block(np.datetime64(ICS[0]))[0]["lat"].values); cosl = np.cos(np.radians(lat))[:, None]
    wn = lambda d: np.sqrt((d ** 2 * cosl).sum())

    names = [nm for nm, _ in OUTVARS]
    sig = {m: np.zeros(len(names)) for m in mechs}; norm = np.zeros(len(names)); nIC = 0
    t0 = time.time()
    for ic in ICS:
        blk = fc.load_block(np.datetime64(ic), nframes=fc.INPUT_WINDOW)
        inp, tgt, frc = fc.build_batch_inputs([blk], 0, tc)
        base = out_fields(apply(inp, tgt, frc, noop)[0])
        nrm = {nm: base[nm].std() + 1e-9 for nm in names}
        for k, nm in enumerate(names): norm[k] += 1.0
        for m in mechs:
            dosed = out_fields(apply(inp, tgt, frc, fc.coef_patch(sae, [feat[m]], +1.0))[0])
            for k, nm in enumerate(names):
                sig[m][k] += wn(dosed[nm] - base[nm]) / nrm[nm]
        print(f"  IC {ic} done {(time.time()-t0)/60:.1f}m", flush=True); nIC += 1
    for m in mechs: sig[m] /= nIC

    print(f"\nMechanism x output signature (Δ per output, std-normalized; row-normalized to max=1):")
    print(f"  {'mechanism':>10} | " + " ".join(f"{nm[:8]:>8}" for nm in names))
    for m in mechs:
        r = sig[m] / (sig[m].max() + 1e-9)
        cells = " ".join(f"{v:>8.2f}" for v in r)
        peak = names[int(np.argmax(sig[m]))]
        print(f"  {m:>10} | {cells}   -> peak: {peak}", flush=True)
    np.save(fc.ROOT / "results/fs_modularity.npy", dict(sig={m: sig[m] for m in mechs}, names=names, feat=feat), allow_pickle=True)
    print("-> results/fs_modularity.npy")

if __name__ == "__main__":
    main()
