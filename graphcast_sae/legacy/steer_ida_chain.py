"""Causal chain into the tropical-cyclone feature during Hurricane Ida.

State-SPECIFIC (Ida present & intensifying, Aug 27-29 2021 — compatible states, so averaging
denoises without washing out the phenomenon). For each localized cast feature we test, at the Ida
state, both directions of intervention and read the effect on every cast feature + the TC feature
3243 at t+6h:
  dose  (coef=+1, double it)  -> sufficiency: does enhancing it intensify the TC feature?
  ablate(coef=-1, remove it)  -> necessity:  does removing it weaken the TC feature?
Then trace the directed cast->...->3243 paths = the interpretable causal chain feeding the cyclone.

Paper: not in the paper; kept for provenance only
Inputs: candidates/fs_feature_catalog.npy (not shipped, see docs/REPRODUCE.md); results/fs_ida_castsel.npy (not shipped, see docs/REPRODUCE.md); GraphCast params (GRAPHCAST_PARAMS)
Outputs: results/fs_ida_chain.npy
Run:   # JAX env, GPU (~46 GB)
    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -m graphcast_sae.legacy.steer_ida_chain
"""
import os, sys, time
os.environ["FS_DEVICE"] = "gpu"; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np, jax.numpy as jnp, xarray as xr, dataclasses
import graphcast_sae.common.fs_common as fc
from graphcast import data_utils, rollout

TC = 3243
ICS = ["2021-08-27", "2021-08-28", "2021-08-29"]              # Ida present & intensifying (compatible)

def numpyify(ds):
    return xr.Dataset({v: (ds[v].dims, np.asarray(ds[v].values)) for v in ds.data_vars},
                      coords={k: ds.coords[k] for k in ds.coords})

def build_io(t0, tc):
    blk, times, statics = fc.load_block(np.datetime64(t0), nframes=4)
    w = blk.assign_coords(time=(times - times[0]).astype("timedelta64[ns]"))
    for v in list(w.data_vars):
        if "time" in w[v].dims: w[v] = w[v].expand_dims("batch")
    for v in fc.STATIC_VARS: w[v] = statics[v]
    w = w.assign_coords(datetime=(("batch", "time"), times[None, :].astype("datetime64[ns]")))
    return data_utils.extract_inputs_targets_forcings(
        w, target_lead_times=slice("6h", "12h"), **dataclasses.asdict(tc))

def main():
    cast = list(np.load(fc.ROOT / "results/fs_ida_castsel.npy", allow_pickle=True).item()["cast"])
    nodes = cast + [TC]; F = len(nodes); tgt_idx = np.array(nodes)
    cat = np.load(fc.ROOT / "candidates/fs_feature_catalog.npy", allow_pickle=True).item()
    lo = np.where(cat["clon"] > 180, cat["clon"] - 360, cat["clon"])
    lab = {fi: f"{fi}({cat['clat'][fi]:+.0f},{lo[fi]:+.0f})" for fi in nodes}
    print(f"cast={len(cast)} + TC {TC}; ICs={ICS}", flush=True)

    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply(mc, tc, stats, sae=sae, bf16=True)
    apply = fc.make_apply(params, rf, patched=True)
    noop = fc.noop_patch(sae)

    def codes_sum(acts):
        X = jnp.asarray(np.asarray(acts, np.float32).reshape(-1, fc.D_IN))
        return np.asarray(sae.codes(X).sum(0))

    def two_step(inp, tgt, frc, patch):
        tct = tgt.time.isel(time=slice(0, 1))
        for c in ("datetime",):
            if c in tgt.coords: tgt = tgt.drop_vars(c)
            if c in frc.coords: frc = frc.drop_vars(c)
        t0 = tgt.isel(time=slice(0, 1)).assign_coords(time=tct); f0 = frc.isel(time=slice(0, 1)).assign_coords(time=tct)
        t1 = tgt.isel(time=slice(1, 2)).assign_coords(time=tct); f1 = frc.isel(time=slice(1, 2)).assign_coords(time=tct)
        p0 = numpyify(apply(inp, t0, f0, patch)[0])
        in1 = rollout._get_next_inputs(inp, xr.merge([p0, f0])).assign_coords(time=inp.coords["time"])
        _, a1 = apply(in1, t1, f1, noop)
        return codes_sum(a1)[tgt_idx]

    # per-IC effect on the TC feature, both directions (also stored to check compatible-state agreement)
    Ad = np.zeros((len(cast), F)); Aa = np.zeros((len(cast), F))
    tc_dose = np.zeros((len(cast), len(ICS))); tc_abl = np.zeros((len(cast), len(ICS)))
    t0 = time.time()
    for k, ic in enumerate(ICS):
        inp, tg, fr = build_io(ic, tc)
        base = two_step(inp, tg, fr, noop)
        for r, fi in enumerate(cast):
            d = two_step(inp, tg, fr, fc.coef_patch(sae, [int(fi)], +1.0)) - base
            a = two_step(inp, tg, fr, fc.coef_patch(sae, [int(fi)], -1.0)) - base
            Ad[r] += d; Aa[r] += a
            tc_dose[r, k] = d[-1]; tc_abl[r, k] = a[-1]        # effect on TC (last index)
        print(f"  IC {ic} done  {(time.time()-t0)/60:.1f}m", flush=True)
    Ad /= len(ICS); Aa /= len(ICS)

    np.save(fc.ROOT / "results/fs_ida_chain.npy",
            dict(Ad=Ad, Aa=Aa, cast=cast, nodes=nodes, labels=lab, ics=ICS,
                 tc_dose=tc_dose, tc_abl=tc_abl), allow_pickle=True)

    # --- causal drivers of the TC feature: dose UP should raise it, ablate should lower it ---
    print(f"\nCausal effect on TC feature {TC} (mean over Ida timesteps):")
    print(f"  {'feature':>14}{'dose+':>9}{'ablate-':>9}{'sign-consistent?':>18}")
    rank = sorted(range(len(cast)), key=lambda r: -(Ad[r, -1] - Aa[r, -1]))
    for r in rank:
        fi = cast[r]
        # causally-responsible driver: dosing raises TC AND ablating lowers it, agreeing across the 3 ICs
        consist = (np.sign(tc_dose[r]).sum() != 0 and (np.sign(tc_dose[r]) == np.sign(tc_dose[r][0])).all()
                   and (np.sign(tc_abl[r]) == np.sign(tc_abl[r][0])).all())
        flag = "DRIVER" if (Ad[r, -1] > 0 and Aa[r, -1] < 0 and consist) else ""
        print(f"  {lab[fi]:>14}{Ad[r,-1]:>+9.2f}{Aa[r,-1]:>+9.2f}{('  yes '+flag) if consist else '  no':>18}", flush=True)

    # --- trace the chain: strongest cast->cast dose edges that feed a driver ---
    print(f"\nStrongest causal links among cast features (dose i -> Δj), the chain structure:")
    edges = sorted([(r, c, Ad[r, c]) for r in range(len(cast)) for c in range(len(cast)) if r != c],
                   key=lambda x: -abs(x[2]))[:10]
    for r, c, w in edges:
        print(f"  {lab[cast[r]]:>14}  ->  {lab[cast[c]]:<14}  {w:+.2f}", flush=True)
    print("-> results/fs_ida_chain.npy")

if __name__ == "__main__":
    main()
