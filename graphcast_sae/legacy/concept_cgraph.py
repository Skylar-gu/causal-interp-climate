"""Interventional causal graph among CONCEPTS (not features, not places).

Implements docs/prereg/prereg_concept_graph.md (CG-1..CG-6), frozen before this file existed. Same doser, same reader, same GAMMA and same ICs as steer_cgraph.py -- only the
aggregation level changes, so CG-1's comparison to the published feature-level numbers
(rho +0.181, sign 0.513, top-20 6/20) is like-for-like.

For source concept i: dose all 15 of its exemplar features at t, advance one 6-h step, read
every concept j's summed activation change vs an undosed baseline.

    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \

Paper: not in the paper; kept for provenance only
Inputs: results/fs_atlas_extra.npy (not shipped, see docs/REPRODUCE.md); GraphCast params (GRAPHCAST_PARAMS)
Outputs: results/fs_concept_cgraph.npy
Run:   # JAX env, GPU (~46 GB)
    python -m graphcast_sae.legacy.concept_cgraph
"""
import os, sys, time
os.environ["FS_DEVICE"] = "gpu"; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np, jax, jax.numpy as jnp, xarray as xr, dataclasses
import graphcast_sae.common.fs_common as fc
from graphcast import data_utils, rollout

GAMMA = 1.0
K = 15                                   # exemplars per concept (prereg: fixed across concepts)
ZMIN = 1.0
SEED = 0
SET_A = ["2018-09-10", "2020-01-05", "2020-04-06", "2020-07-06", "2020-10-05", "2019-06-15"]
SET_B = ["2020-02-20", "2020-05-25", "2020-08-15", "2020-11-20"]
OUT = os.environ.get("CC_OUT", "results/fs_concept_cgraph.npy")
# div250 EXCLUDED in the prereg: 0 features at argmax. R1 is struck, not scored.
CONCEPTS = ["vort850", "q600", "ascent", "shear", "t850", "z500", "jet250",
            "blocking", "atm_river", "baroclinicity"]

def build_concepts():
    a = np.load(fc.ROOT / "results/fs_atlas_extra.npy", allow_pickle=True).item()
    z, refs, ze, ne = a["z"], a["node_refs"], a["z_extra"], a["node_extra"]
    alive = a["zcnt"] > 300

    def zc(n):
        return z[:, refs.index(n)] if n in refs else ze[:, ne.index(n)]

    Z = np.stack([zc(n) for n in CONCEPTS], 1)
    lab, mx = np.argmax(np.abs(Z), 1), np.abs(Z).max(1)
    groups = {}
    for k, n in enumerate(CONCEPTS):
        m = alive & (lab == k) & (mx > ZMIN)
        idx = np.where(m)[0]
        idx = idx[np.argsort(-np.abs(Z[idx, k]))][:K]
        groups[n] = idx
        print(f"  {n:<14} {len(idx):>3} exemplars (of {int(m.sum())} qualifying)", flush=True)
    return groups, Z

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
    print("CONCEPT CAUSAL GRAPH — prereg docs/prereg/prereg_concept_graph.md (frozen ec90ea7)\n")
    groups, _ = build_concepts()
    names = list(groups)
    C = len(names)
    members = [groups[n] for n in names]

    # CG-2 NEG: same 150 features, random re-partition into 10 groups of 15
    rng = np.random.default_rng(SEED)
    pool = np.concatenate(members); perm = rng.permutation(pool)
    perm_groups = [perm[i * K:(i + 1) * K] for i in range(C)]
    print(f"\nconcept_perm NEG control: {len(pool)} features re-partitioned into {C} x {K}")

    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply(mc, tc, stats, sae=sae, bf16=True)
    apply = fc.make_apply(params, rf, patched=True)
    noop = fc.noop_patch(sae)

    def codes_sum(acts):
        X = jnp.asarray(np.asarray(acts, np.float32).reshape(-1, fc.D_IN))
        return np.asarray(sae.codes(X).sum(0))

    def step(inp, tgt1, frc1, patch):
        preds, acts = apply(inp, tgt1, frc1, patch)
        return numpyify(preds), acts

    ICS = SET_A + SET_B
    A_ic = np.zeros((len(ICS), C, C))          # real concepts, per IC
    P_ic = np.zeros((len(ICS), C, C))          # concept_perm NEG, per IC
    t0 = time.time()
    for w, ic in enumerate(ICS):
        inp, tgt, frc = build_io(ic, tc)
        tct = tgt.time.isel(time=slice(0, 1))
        for c in ("datetime",):
            if c in tgt.coords: tgt = tgt.drop_vars(c)
            if c in frc.coords: frc = frc.drop_vars(c)
        tgt0 = tgt.isel(time=slice(0, 1)).assign_coords(time=tct)
        frc0 = frc.isel(time=slice(0, 1)).assign_coords(time=tct)
        tgt1 = tgt.isel(time=slice(1, 2)).assign_coords(time=tct)
        frc1 = frc.isel(time=slice(1, 2)).assign_coords(time=tct)

        p0b, _ = step(inp, tgt0, frc0, noop)
        in1b = rollout._get_next_inputs(inp, xr.merge([p0b, frc0])).assign_coords(time=inp.coords["time"])
        _, a1b = step(in1b, tgt1, frc1, noop)
        base = codes_sum(a1b)

        for tag, grps, store in (("real", members, A_ic), ("perm", perm_groups, P_ic)):
            for r, feats in enumerate(grps):
                p0, _ = step(inp, tgt0, frc0, fc.coef_patch(sae, [int(f) for f in feats], GAMMA))
                in1 = rollout._get_next_inputs(inp, xr.merge([p0, frc0])).assign_coords(time=inp.coords["time"])
                _, a1 = step(in1, tgt1, frc1, noop)
                d = codes_sum(a1) - base
                store[w, r] = [d[g].sum() for g in grps]
        el = (time.time() - t0) / 60
        print(f"  IC {ic} done ({w+1}/{len(ICS)})  {el:.1f}m  eta {el/(w+1)*(len(ICS)-w-1):.0f}m", flush=True)

    np.save(fc.ROOT / OUT, dict(A_ic=A_ic, P_ic=P_ic, names=names,
                                members=[list(map(int, m)) for m in members],
                                perm=[list(map(int, m)) for m in perm_groups],
                                set_a=SET_A, set_b=SET_B, ics=ICS, gamma=GAMMA, K=K,
                                prereg="CG-1..CG-6, docs/prereg/prereg_concept_graph.md @ ec90ea7"),
            allow_pickle=True)
    print(f"\n-> {OUT}   (score with graphcast_sae/legacy/concept_cgraph_score.py)")

if __name__ == "__main__":
    main()
