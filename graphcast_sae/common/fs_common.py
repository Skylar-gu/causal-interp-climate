"""Shared machinery for the FLAGSHIP (0.25°/37-lev) SAE suite.

Env: the JAX environment (requirements.txt, "GPU experiments" block; see also
docs/prereg/prereg_flagship_g2_suite.md). Forward passes run on CPU by default
(`JAX_PLATFORMS=cpu`): one flagship forward needs ~32 GB of activation memory.
Set FS_DEVICE=gpu on a card with >= 40 GB.

What lives here
---------------
* flagship checkpoint + normalization stats loading
* the layer-8 hook. **HOOK_STEP = 9** (1-indexed `_process_step` count) — the
  authors' `layer0008_*` files are 0-indexed, so their layer 8 is our 9th step.
  Verified: FVU 0.132 at step 9 vs 0.194 at step 8 and 0.508 at step 16
* the published TopK SAE re-implemented in jnp, with the authors' semantics:
  the encoder sees a per-token centred + unit-L2-normalized input, but the decoder
  output is in **RAW activation units** (their training loss compared `recon` to the
  un-normalized input — this is why FVU must be scored against raw x, and why the
  patch algebra below has no global `scale` factor unlike the small-model version)
* the intervention patch, threaded as RUNTIME arrays so every arm shares ONE
  compiled graph (small-model lesson: per-arm trace constants let XLA fuse arms
  differently and a *zero* patch moved 2m temperature by 0.16 K)
* WeatherBench-2 0.25°/37-lev streaming in the authors' exact window semantics
* the forecast-space scorer (stddev-normalized, cos-lat-weighted nRMSE).

Paper: shared: model, SAE, patching, WeatherBench2 streaming
Inputs: results/... (not shipped, see docs/REPRODUCE.md); GraphCast params (GRAPHCAST_PARAMS); WeatherBench2 ERA5 (GCS, streamed); i.i.d. dump (GC_SCRATCH/fs_iid_dump.npy, extraction/extract_iid_dump.py); $GC_SCRATCH/fs_acts (extraction/fs_extract.py)
Outputs: $GC_SCRATCH/flagship_m6_degree.npy (M6 node-degree cache, built on first use); everything else is returned to the caller
Run:   # JAX env, GPU (~46 GB)
    python -m graphcast_sae.common.fs_common
"""
import dataclasses
import functools
import os
import pathlib

os.environ.setdefault("JAX_PLATFORMS", "cpu" if os.environ.get("FS_DEVICE", "cpu") == "cpu" else "")

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from graphcast import (casting, checkpoint, data_utils, graphcast as gc,
                       icosahedral_mesh as im, normalization)

from graphcast_sae import paths as _paths      # the one place locations are resolved
ROOT = _paths.REPO_ROOT          # re-exported: `fc.ROOT / "results/..."` is this repo
ASSETS = _paths.ASSETS           # DeepMind release stats/ (GRAPHCAST_ASSETS)
WEIGHTS = _paths.WEIGHTS
CKPT = _paths.GRAPHCAST_PARAMS   # DeepMind 0.25 deg / 37-level checkpoint (GRAPHCAST_PARAMS)
SAE_PT = _paths.SAE_WEIGHTS      # the shipped .npz; the name is historical (SAEJax reads both)
SCRATCH = _paths.SCRATCH         # regenerable dumps (GC_SCRATCH)
ACTS_DIR = _paths.ACTS_DIR
DEG_NPY = _paths.MESH_DEGREE
MESH_GEOM = _paths.MESH_GEOM     # shipped: data/mesh_2to6_geom.npy
IID_DUMP, IID_META = _paths.IID_DUMP, _paths.IID_META

ZARR = _paths.WB2_ERA5_ZARR
HOOK_STEP = 9                      # 1-indexed; == authors' 0-indexed "layer0008"
N_MESH = 40962                     # M6 merged multi-mesh nodes
D_IN = 512
INPUT_WINDOW = 3
STEP = np.timedelta64(6, "h")
CHUNK = 4096                       # tokens per SAE chunk inside the jitted forward

SURFACE_VARS = ("2m_temperature", "mean_sea_level_pressure",
                "10m_v_component_of_wind", "10m_u_component_of_wind",
                "total_precipitation_6hr")
ATMOS_VARS = ("temperature", "geopotential", "u_component_of_wind",
              "v_component_of_wind", "vertical_velocity", "specific_humidity")
STATIC_VARS = ("geopotential_at_surface", "land_sea_mask")
FORCING_VARS = ("toa_incident_solar_radiation",)
PROG = list(SURFACE_VARS) + list(ATMOS_VARS)

# --------------------------------------------------------------- model -----
def load_model():
    """flagship params + configs + the stats the model normalizes with."""
    with open(CKPT, "rb") as fh:
        ck = checkpoint.load(fh, gc.CheckPoint)
    stats = {n: xr.load_dataset(ASSETS / f"stats/{n}.nc").compute() for n in
             ("diffs_stddev_by_level", "mean_by_level", "stddev_by_level")}
    return ck.params, ck.model_config, ck.task_config, stats

# ----------------------------------------------------------------- SAE -----
class SAEJax:
    """The published TopK SAE (k=32, dict 4096) in jnp, authors' semantics.

        xn    = (x - mean_tok(x)) / ||x - mean_tok(x)||     per-token normalization
        f     = TopK_k(relu((xn - b_pre) @ W_enc.T))        sparse code
        recon = f @ W_dec_unit.T + b_pre                    RAW activation units
    """

    def __init__(self, path=SAE_PT):
        npz = pathlib.Path(str(path)).with_suffix(".npz")
        if npz.exists():                                             # torch-free path (JAX env)
            z = np.load(npz); self.tokens = 0
            self.W_enc = jnp.asarray(z["W_enc"]); self.W_dec = jnp.asarray(z["W_dec"])
            self.b_pre = jnp.asarray(z["b_pre"])
            self.k = 32; self.n_features = int(self.W_enc.shape[0]); return
        import torch
        sd = torch.load(path, map_location="cpu")
        self.tokens = int(sd.get("tokens", 0))
        sd = sd["model_state"] if "model_state" in sd else sd
        Wdec = sd["dec.weight"].float().numpy()                       # (512, F)
        Wdec = Wdec / np.maximum(np.linalg.norm(Wdec, axis=0, keepdims=True), 1e-8)
        self.W_enc = jnp.asarray(sd["enc.weight"].float().numpy())    # (F, 512)
        self.W_dec = jnp.asarray(Wdec)                                # (512, F)
        self.b_pre = jnp.asarray(sd["b_pre"].float().numpy())         # (512,)
        self.k = 32
        self.n_features = int(self.W_enc.shape[0])

    def norm_tok(self, x):
        xn = x - x.mean(-1, keepdims=True)
        return xn / jnp.maximum(jnp.linalg.norm(xn, axis=-1, keepdims=True), 1e-6)

    def codes(self, x):
        """x (n,512) RAW -> dense TopK code (n,F)."""
        pre = jax.nn.relu((self.norm_tok(x) - self.b_pre) @ self.W_enc.T)
        vals, idx = jax.lax.top_k(pre, self.k)
        f = jnp.zeros_like(pre)
        rows = jnp.arange(x.shape[0])[:, None]
        return f.at[rows, idx].set(vals)

    def delta(self, x, coef, rho, uvec):
        """Patch delta in RAW activation units for one chunk x (n,512).

        coef (F,) per-feature multiplier on that feature's OWN decoder contribution
             (-1 on a set = error-preserving ablation; +alpha on one = dose steering)
        rho  scalar, 1.0 = full SAE substitution (the SAE's own FVU, in forecast space)
        uvec (512,) spatially uniform shift at every node.
        coef=0, rho=0, uvec=0 reproduces the model exactly.
        """
        f = self.codes(x)
        recon = f @ self.W_dec.T + self.b_pre
        d = (f * coef) @ self.W_dec.T
        d = d + rho * (recon - x)
        return d + uvec[None, :]

    def delta_cond(self, x, fsel, ftarget, nmask):
        """Counterfactual-conditioning delta: inside the nmask nodes, CAP the selected features
        (fsel>0) at a normal level ftarget — stripping the anomalous excess while keeping the
        background. This is 'restore convection to normal, locally', not 'delete convection'.
        fsel (F,) 0/1; ftarget (F,) per-feature normal level; nmask (n,) 0/1 node mask."""
        f = self.codes(x)
        capped = jnp.where(fsel[None, :] > 0, jnp.minimum(f, ftarget[None, :]), f)
        d = (capped - f) @ self.W_dec.T                 # nonzero only where selected features exceed normal
        return nmask[:, None] * d

    def delta_gain(self, x, fsel, ftarget, nmask, gain):
        """SCALE the anomalous excess by `gain` instead of removing it (added 2026-08-17).

        delta_cond answers "what if this mechanism were only normal here". It can only
        ever REDUCE activation (jnp.minimum), so it cannot ask the other half of the
        causal question: what if there were MORE of it. This generalizes it to a
        one-parameter family on the excess above normal,

            f -> f + (gain - 1) * max(f - ftarget, 0)

        so gain=0 is EXACTLY delta_cond (verified numerically), gain=1 is the identity
        (a no-op, i.e. baseline), and gain>1 amplifies. One instrument, one knob, with
        the committed ablation arm sitting at one end of it -- which also means g=0
        reproducing the published number is a built-in regression check.

        Amplification is off-distribution by construction: the model never saw a state
        with 3x the normal convective excess, so a monotone response is evidence and a
        non-monotone or exploding one is the instrument's limit, not the model's physics.
        Report the whole curve, never one gain."""
        f = self.codes(x)
        excess = jnp.maximum(f - ftarget[None, :], 0.0)
        scaled = jnp.where(fsel[None, :] > 0, f + (gain - 1.0) * excess, f)
        d = (scaled - f) @ self.W_dec.T
        return nmask[:, None] * d

    def delta_cond_freeze(self, x, fsel, ftarget, nmask, jidx, fref, jmask):
        """MEDIATION patch (added 2026-08-20): ablate i AND clamp j to its BASELINE value.

        delta_cond and delta_gain are both one-sided caps toward a *constant* ftarget.
        A path test needs the other thing: hold the mediator at the value it took in the
        unperturbed run, at this node, at this step -- a TWO-SIDED clamp to a time-varying,
        per-node reference. That is do(i) vs do(i, j = its natural value), i.e. the
        controlled direct effect, and if the effect collapses the pathway runs through j.

            f'      = f + nmask * (min(f, ftarget)|_fsel - f)      the ablation, as delta_cond
            f'[:,j] = jmask * fref + (1 - jmask) * f'[:,j]         the clamp
            delta   = (f' - f) @ W_dec.T

        jidx  (J,)          int, the frozen feature ids
        fref  (n_nodes, J)  their baseline codes AT THIS ROLLOUT STEP (the caller indexes
                            the step; a length-H object meeting an (n_nodes, F) array is
                            an earlier indexing bug that killed every ramp arm)
        jmask (n_nodes, J)  0/1, which (node, frozen-feature) pairs are actually clamped.
                            All zeros is an EXACT no-op: 0*fref + 1*f' = f' in IEEE, and
                            (f'-f) is then exactly the delta_cond delta. That is what lets
                            the mediation battery's BASELINE arm carry the same 6-tuple
                            pytree as the freeze arms, so every arm shares ONE compiled
                            graph (the small-model lesson at the top of this file).

        EXACTNESS. With fsel=0 and jmask=1 on a run whose fref came from the baseline arm
        of the SAME process, f'[:,j] == fref == f[:,j] elementwise, so f'-f is exactly the
        zero matrix and delta is exactly zero. The clamp is therefore a no-op on the
        baseline trajectory by construction; any residual is the GPU nondeterminism floor
        (docs/notes/nondeterminism_floor_2026_08_20.md), and measuring it against a jmask=0 arm
        of the same graph is the correctness test that must pass before any mediation
        number is quoted."""
        f = self.codes(x)
        capped = jnp.where(fsel[None, :] > 0, jnp.minimum(f, ftarget[None, :]), f)
        fnew = f + nmask[:, None] * (capped - f)
        cur = fnew[:, jidx]
        fnew = fnew.at[:, jidx].set(jmask * fref + (1.0 - jmask) * cur)
        return (fnew - f) @ self.W_dec.T

def sae_numpy(path=SAE_PT):
    """Same weights as SAEJax but plain numpy — for the CPU catalog/analysis half."""
    path = pathlib.Path(str(path))
    if path.suffix == ".npz":                                        # the shipped weights
        z = np.load(path)
        return dict(W_enc=np.asarray(z["W_enc"], np.float32), W_dec=np.asarray(z["W_dec"], np.float32),
                    b_pre=np.asarray(z["b_pre"], np.float32), k=32)
    import torch                                                     # authors' .pt checkpoint
    sd = torch.load(path, map_location="cpu")
    sd = sd["model_state"] if "model_state" in sd else sd
    Wdec = sd["dec.weight"].float().numpy()
    Wdec = Wdec / np.maximum(np.linalg.norm(Wdec, axis=0, keepdims=True), 1e-8)
    return dict(W_enc=sd["enc.weight"].float().numpy(), W_dec=Wdec,
                b_pre=sd["b_pre"].float().numpy(), k=32)

def encode_np(X, sae, want_recon=False):
    """X (n,512) raw float32 -> (code (n,F) float32, recon or None). Numpy/torch."""
    import torch
    x = torch.from_numpy(np.asarray(X, np.float32))
    xn = x - x.mean(1, keepdim=True)
    xn = xn / xn.norm(dim=1, keepdim=True).clamp_min(1e-6)
    pre = torch.relu((xn - torch.from_numpy(sae["b_pre"])) @ torch.from_numpy(sae["W_enc"]).T)
    v, i = torch.topk(pre, sae["k"], dim=1)
    z = torch.zeros_like(pre).scatter_(1, i, v)
    recon = (z @ torch.from_numpy(sae["W_dec"]).T + torch.from_numpy(sae["b_pre"])) \
        if want_recon else None
    return z.numpy(), (recon.numpy() if want_recon else None)

def fvu_raw(X, recon):
    """FVU scored against the RAW activation — the correct target for this SAE."""
    X = np.asarray(X, np.float64); R = np.asarray(recon, np.float64)
    return float(((X - R) ** 2).sum() / ((X - X.mean(0, keepdims=True)) ** 2).sum())

# ------------------------------------------------------------- forward -----
def chunked(fn, X, chunk=CHUNK):
    N, d = X.shape
    pad = (-N) % chunk
    Xp = jnp.pad(X, ((0, pad), (0, 0)))
    Y = jax.lax.map(fn, Xp.reshape(-1, chunk, d))
    return Y.reshape(-1, Y.shape[-1])[:N]

def build_apply(model_config, task_config, stats, sae=None, bf16=False):
    """Return (run_forward, captured).

    sae=None  -> capture-only forward (no patch argument), used by extraction.
    sae given -> patched forward; `patch = (coef (F,), rho (), uvec (512,))` are
    RUNTIME arrays so all arms share one compiled graph.
    float32 by default: the small-model plumbing check measured bf16 repeat-call
    noise at 1.35e-2 K vs 8.8e-4 K in float32, the same order as real effects.
    """
    def wrapped():
        predictor = gc.GraphCast(model_config, task_config)
        if bf16:
            predictor = casting.Bfloat16Cast(predictor)
        return normalization.InputsAndResiduals(
            predictor, diffs_stddev_by_level=stats["diffs_stddev_by_level"],
            mean_by_level=stats["mean_by_level"],
            stddev_by_level=stats["stddev_by_level"])

    captured = {"count": 0, "acts": {}, "patch": None}

    def interceptor(next_fun, args, kwargs, context):
        out = next_fun(*args, **kwargs)
        if (context.method_name == "_process_step"
                and context.module.module_name.split("/")[-1] == "mesh_gnn"):
            captured["count"] += 1
            if captured["count"] == HOOK_STEP:
                ns = out.nodes["mesh_nodes"]
                feats = ns.features
                shp = feats.shape
                X = feats.reshape(-1, shp[-1]).astype(jnp.float32)
                if sae is not None:
                    coef, rho, uvec = captured["patch"]
                    X = X + chunked(lambda c: sae.delta(c, coef, rho, uvec), X)
                    new = X.reshape(shp).astype(feats.dtype)
                    out = out._replace(nodes={**out.nodes,
                                              "mesh_nodes": ns._replace(features=new)})
                captured["acts"][HOOK_STEP] = X.reshape(shp)
        return out

    if sae is None:
        @hk.transform_with_state
        def run_forward(inputs, targets_template, forcings):
            captured["count"] = 0
            with hk.intercept_methods(interceptor):
                preds = wrapped()(inputs, targets_template=targets_template,
                                  forcings=forcings)
            return preds, captured["acts"][HOOK_STEP]
    else:
        @hk.transform_with_state
        def run_forward(inputs, targets_template, forcings, patch):
            captured["count"] = 0
            captured["patch"] = patch
            with hk.intercept_methods(interceptor):
                preds = wrapped()(inputs, targets_template=targets_template,
                                  forcings=forcings)
            return preds, captured["acts"][HOOK_STEP]
    return run_forward, captured

def build_apply_cond(model_config, task_config, stats, sae, bf16=False):
    """Like build_apply, but the patch CONDITIONS features to a normal level within a node mask
    (counterfactual 'restore to normal, localized', via sae.delta_cond).
    patch = (fsel (F,), ftarget (F,), nmask (n_mesh,)) -> delta_cond, or
    (fsel, ftarget, nmask, gain) -> delta_gain, or
    (fsel, ftarget, nmask, jidx (J,), fref (n_mesh,J), jmask (n_mesh,J)) -> delta_cond_freeze,
    the mediation clamp. Runs on the full node set (no chunk)."""
    def wrapped():
        predictor = gc.GraphCast(model_config, task_config)
        if bf16:
            predictor = casting.Bfloat16Cast(predictor)
        return normalization.InputsAndResiduals(
            predictor, diffs_stddev_by_level=stats["diffs_stddev_by_level"],
            mean_by_level=stats["mean_by_level"], stddev_by_level=stats["stddev_by_level"])

    captured = {"count": 0, "acts": {}, "patch": None}

    def interceptor(next_fun, args, kwargs, context):
        out = next_fun(*args, **kwargs)
        if (context.method_name == "_process_step"
                and context.module.module_name.split("/")[-1] == "mesh_gnn"):
            captured["count"] += 1
            if captured["count"] == HOOK_STEP:
                ns = out.nodes["mesh_nodes"]; feats = ns.features; shp = feats.shape
                X = feats.reshape(-1, shp[-1]).astype(jnp.float32)
                p = captured["patch"]
                fsel, ftarget, nmask = p[0], p[1], p[2]
                # a 4-tuple carries a gain and selects delta_gain; 3-tuples keep the
                # exact previous behaviour, so every committed arm is untouched.
                # a 6-tuple carries (jidx, fref, jmask) and selects the mediation clamp
                # (added 2026-08-20); the two branches above are byte-for-byte as before.
                if len(p) == 3:
                    _d = sae.delta_cond(X, fsel, ftarget, nmask)
                elif len(p) == 4:
                    _d = sae.delta_gain(X, fsel, ftarget, nmask, p[3])
                else:
                    _d = sae.delta_cond_freeze(X, fsel, ftarget, nmask, p[3], p[4], p[5])
                X = X + _d
                new = X.reshape(shp).astype(feats.dtype)
                out = out._replace(nodes={**out.nodes, "mesh_nodes": ns._replace(features=new)})
                captured["acts"][HOOK_STEP] = X.reshape(shp)
        return out

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings, patch):
        captured["count"] = 0; captured["patch"] = patch
        with hk.intercept_methods(interceptor):
            preds = wrapped()(inputs, targets_template=targets_template, forcings=forcings)
        return preds, captured["acts"][HOOK_STEP]
    return run_forward, captured

def make_apply(params, run_forward, patched=True):
    fn = functools.partial(run_forward.apply, params, {}, jax.random.PRNGKey(0))
    if patched:
        return jax.jit(lambda inp, tgt, frc, patch: fn(inp, tgt, frc, patch)[0])
    return jax.jit(lambda inp, tgt, frc: fn(inp, tgt, frc)[0])

def noop_patch(sae):
    return (jnp.zeros(sae.n_features, jnp.float32), jnp.float32(0.0),
            jnp.zeros(D_IN, jnp.float32))

def coef_patch(sae, feats, value):
    c = np.zeros(sae.n_features, np.float32)
    c[np.asarray(feats, int)] = value
    return (jnp.asarray(c), jnp.float32(0.0), jnp.zeros(D_IN, jnp.float32))

def recon_patch(sae):
    return (jnp.zeros(sae.n_features, jnp.float32), jnp.float32(1.0),
            jnp.zeros(D_IN, jnp.float32))

# ---------------------------------------------------------------- data -----
@functools.lru_cache(maxsize=1)
def open_wb2():
    """The authors' own training source: WB2 0.25°/37-lev derived zarr (anon GCS)."""
    import gcsfs
    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(ZARR[5:]), consolidated=True)
    ren = {}
    if "latitude" in ds.coords:
        ren["latitude"] = "lat"
    if "longitude" in ds.coords:
        ren["longitude"] = "lon"
    if ren:
        ds = ds.rename(ren)
    if ds.lat[0] > ds.lat[-1]:
        ds = ds.reindex(lat=ds.lat[::-1])
    keep = list(SURFACE_VARS) + list(ATMOS_VARS) + list(FORCING_VARS) + list(STATIC_VARS)
    ds = ds[[v for v in keep if v in ds.data_vars]]
    statics = ds[list(STATIC_VARS)].load()
    return ds, statics

def seasonal_starts(n=24, year=2021):
    """n window centres evenly spaced across `year`, snapped to the 6-h grid.

    Fixed by rule (prereg §1) so the sample cannot be tuned after the fact.
    """
    t0 = np.datetime64(f"{year}-01-01T00")
    span = np.timedelta64(365 * 24, "h")
    out = []
    for i in range(n):
        h = int(round(float(span / np.timedelta64(1, "h")) * i / n / 6.0)) * 6
        out.append(t0 + np.timedelta64(h, "h"))
    return np.array(out, dtype="datetime64[ns]")

def load_block(center, nframes=INPUT_WINDOW):
    """One window: `nframes` consecutive 6-h frames starting at center-6h.

    Authors' semantics (three_step_window): times [t-6h, t, t+6h, ...], a `batch`
    dim on every time-dependent var, `time` rebased to a timedelta from the FIRST
    frame, absolute stamps on `datetime`.
    """
    ds, statics = open_wb2()
    times = np.datetime64(center) - STEP + np.arange(nframes) * STEP
    blk = ds[list(SURFACE_VARS) + list(ATMOS_VARS) + list(FORCING_VARS)].sel(time=times).load()
    return blk, times, statics

def build_batch_inputs(blocks, s, task_config):
    """Batch the s-th 3-frame sub-window of each block into GraphCast inputs."""
    wins = []
    for blk, times, _ in blocks:
        at = times[s:s + INPUT_WINDOW]
        w = blk.sel(time=at)
        w = w.assign_coords(time=(at - at[0]).astype("timedelta64[ns]"))
        for v in list(w.data_vars):
            if "time" in w[v].dims:
                w[v] = w[v].expand_dims("batch")
        wins.append(w)
    big = xr.concat(wins, dim="batch", data_vars="all") if len(wins) > 1 else wins[0]
    for v in STATIC_VARS:
        big[v] = blocks[0][2][v]
    dts = np.stack([t[s:s + INPUT_WINDOW] for _, t, _ in blocks]).astype("datetime64[ns]")
    big = big.assign_coords(datetime=(("batch", "time"), dts))
    return data_utils.extract_inputs_targets_forcings(
        big, target_lead_times=slice("6h", "6h"), **dataclasses.asdict(task_config))

# -------------------------------------------------------------- scoring ----
class Scorer:
    """stddev-normalized, cos-lat-weighted RMSE over all prognostic vars/levels."""

    def __init__(self, stats, sample_preds):
        sd = stats["stddev_by_level"]
        levels = np.asarray(sample_preds["level"].values)
        self.sig = {}
        for v in PROG:
            s = sd[v]
            self.sig[v] = (np.asarray(s.sel(level=levels).values, np.float64)
                           if "level" in s.dims else float(s.values))
        lat = np.asarray(sample_preds["lat"].values, np.float64)
        self.w = np.cos(np.deg2rad(lat))
        self.dims = {v: sample_preds[v].dims for v in PROG}

    def __call__(self, dy):
        num = den = None
        for v in PROG:
            dims = self.dims[v]
            ax = {k: dims.index(k) for k in dims}
            order = [ax["batch"], ax["time"]] + \
                    ([ax["level"]] if "level" in ax else []) + [ax["lat"], ax["lon"]]
            a = np.transpose(dy[v], order)[:, 0]
            if a.ndim == 3:
                a = a[:, None]
            sig = self.sig[v]
            sig = (np.asarray(sig, np.float64).reshape(1, -1, 1, 1)
                   if np.ndim(sig) else np.full((1, 1, 1, 1), sig))
            z = (a / sig) ** 2 * self.w[None, None, :, None]
            n = z.sum(axis=(1, 2, 3))
            dcount = a.shape[1] * self.w.sum() * a.shape[3]
            num = n if num is None else num + n
            den = dcount if den is None else den + dcount
        return np.sqrt(num / den)

# ----------------------------------------------------------- mesh degree ---
def mesh_degree():
    """Merged M6 multi-mesh node degree, aligned to mesh-node order."""
    if DEG_NPY.exists():
        d = np.load(DEG_NPY, allow_pickle=True)
        d = d.item() if d.dtype == object else d
        deg = d["deg"] if isinstance(d, dict) else d
        return np.asarray(deg, np.float64)
    meshes = im.get_hierarchy_of_triangular_meshes_for_sphere(splits=6)
    merged = im.merge_meshes(meshes)
    faces = np.asarray(merged.faces)
    deg = np.zeros(int(faces.max()) + 1, np.int64)
    edges = set()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add((min(u, v), max(u, v)))
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    np.save(DEG_NPY, dict(deg=deg), allow_pickle=True)
    return deg.astype(np.float64)

def act_path(center):
    ts = np.datetime_as_string(np.datetime64(center, "h"), unit="h")
    return ACTS_DIR / f"layer0008_mesh_gnn_post_res_nodes_mesh_nodes_t{ts}.npy"
