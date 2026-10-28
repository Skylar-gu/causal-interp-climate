"""ERA5-forced ("perfect forecast") reference row for the Ida cyclone-feature progression.

Same IC / box / mesh / model / SAE as ida_mechmaps_prog.py, but the rollout is
TEACHER-FORCED: at every 6-h step the model is fed the true ERA5 state (the target
frames that extract_inputs_targets_forcings already carved out of the loaded block)
instead of its own prediction, so there is no error accumulation. Layer-8 activations
are still read from the model's forward pass -- this is "what the internal cyclone
feature f3243 would look like over the box if the 48-h forecast tracked reality exactly".

Also re-rolls the free-running baseline here so the two curves are strictly comparable
(same code path, same session). Writes tc_truth / tc_truth_mid / tc_freecheck into
results/fs_ida_mechmaps_prog<TAG>.npy in place; every existing key is preserved.

Paper: Fig. 3 ERA5-forced row and its +48 h value 213 (figures/paper_fig_ida_dialup.py reads tc_truth).
The shipped results/fs_ida_mechmaps_prog_v2.npy predates this run and has no tc_truth key.
Run:   # JAX env, GPU (~46 GB)
    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -m graphcast_sae.storms.ida_mechmaps_truth
"""
import os, sys, time
os.environ["FS_DEVICE"] = "gpu"; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np, jax.numpy as jnp, xarray as xr, dataclasses
import graphcast_sae.common.fs_common as fc
from graphcast import data_utils, rollout

TC = 3243; IC = "2021-08-26"; H = 8
REG = dict(lat=(0, 40), lon=(-110, -40))
CAP = [1, 3, 5, 7]                       # capture leads: (h+1)*6 h = +12/+24/+36/+48
TAG = os.environ.get("MECHMAPS_TAG", "_v2")
OUT = fc.ROOT / f"results/fs_ida_mechmaps_prog{TAG}.npy"


def numpyify(ds):
    return xr.Dataset({v: (ds[v].dims, np.asarray(ds[v].values)) for v in ds.data_vars},
                      coords={k: ds.coords[k] for k in ds.coords})


def main():
    geom = np.load(fc.MESH_GEOM, allow_pickle=True).item()
    mlat = geom["lat"]; mlon = np.where(geom["lon"] > 180, geom["lon"] - 360, geom["lon"])
    node = ((mlat >= REG["lat"][0]) & (mlat <= REG["lat"][1])
            & (mlon >= REG["lon"][0]) & (mlon <= REG["lon"][1]))

    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply(mc, tc, stats, sae=sae, bf16=True)
    apply = fc.make_apply(params, rf, patched=True)
    noop = fc.noop_patch(sae)

    blk, times, st = fc.load_block(np.datetime64(IC), nframes=2 + H)
    w = blk.assign_coords(time=(times - times[0]).astype("timedelta64[ns]"))
    for v in list(w.data_vars):
        if "time" in w[v].dims: w[v] = w[v].expand_dims("batch")
    for v in fc.STATIC_VARS: w[v] = st[v]
    w = w.assign_coords(datetime=(("batch", "time"), times[None, :].astype("datetime64[ns]")))
    inp, tgt, frc = data_utils.extract_inputs_targets_forcings(
        w, target_lead_times=slice("6h", f"{6*H}h"), **dataclasses.asdict(tc))
    tct = tgt.time.isel(time=slice(0, 1))
    for c in ("datetime",):
        if c in tgt.coords: tgt = tgt.drop_vars(c)
        if c in frc.coords: frc = frc.drop_vars(c)

    def roll(force_truth):
        """Return {lead_h: f3243 node map over the box} for leads in CAP, and the +30 h map."""
        cur = inp; frames = {}; mid = None
        for h in range(H):
            t0 = time.time()
            ct = tgt.isel(time=slice(h, h + 1)).assign_coords(time=tct)
            cf = frc.isel(time=slice(h, h + 1)).assign_coords(time=tct)
            p, a = apply(cur, ct, cf, noop)                       # activations from the true forward pass
            X = jnp.asarray(np.asarray(a, np.float32).reshape(-1, fc.D_IN))
            code = np.asarray(sae.codes(X))
            fr = code[node, TC]
            if h == 4: mid = fr
            if h in CAP: frames[(h + 1) * 6] = fr
            nxt = numpyify(ct) if force_truth else numpyify(p)    # feed truth vs own prediction
            if h < H - 1:
                cur = rollout._get_next_inputs(
                    cur, xr.merge([nxt, cf])).assign_coords(time=cur.coords["time"])
            print(f"    {'truth' if force_truth else 'free '} step {h+1}/{H}  {time.time()-t0:.0f}s",
                  flush=True)
        return frames, mid

    t0 = time.time()
    tf, tmid = roll(True)
    ff, _ = roll(False)
    print(f"both rollouts done {time.time()-t0:.0f}s", flush=True)

    s_t = {lead: float(np.sum(tf[lead])) for lead in sorted(tf)}
    s_f = {lead: float(np.sum(ff[lead])) for lead in sorted(ff)}
    print("  lead |  ERA5-forced | free-running | forecast err vs truth")
    for lead in sorted(tf):
        print(f"  +{lead:>3} | {s_t[lead]:>11.0f} | {s_f[lead]:>12.0f} | "
              f"{100*(s_f[lead]-s_t[lead])/s_t[lead]:+.0f}%")

    out = np.load(OUT, allow_pickle=True).item()
    out["tc_truth"] = {lead: tf[lead] for lead in sorted(tf)}
    out["tc_truth_mid"] = tmid
    out["tc_freecheck"] = {lead: ff[lead] for lead in sorted(ff)}
    np.save(OUT, out, allow_pickle=True)
    print(f"-> {OUT}  (+ tc_truth, tc_truth_mid, tc_freecheck)", flush=True)


if __name__ == "__main__":
    main()
