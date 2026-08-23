"""Convection-skill-necessity: per-storm counterfactual rollouts (GPU, bf16).

For each storm (from skill_conv_storms.STORMS, ERA5-verified deepeners + one non-developer),
roll GraphCast from the formation IC to +96h under FOUR arms sharing ONE compiled graph
(runtime patch arrays; build_apply_cond / delta_cond, the 'restore to normal within a 1500 km
disk' counterfactual from steer_ida_counterfactual.py):

  baseline    : untouched
  conv-normal : convection features [2401,2067,3174] capped at NORMAL in the disk (honest CF)
  conv-zero   : convection features capped at 0 in the disk (delete, for continuity)
  rand-normal : matched-firing-rate RANDOM control features capped at NORMAL in the disk

Records per lead: (1) INTERNAL TC feature 3243 in the storm box, plus convection 2401 &
control activations (co-location / firing checks); (2) PHYSICAL min MSLP & max 10 m wind in
the box. ERA5 truth for skill is in era5_truth.npy (skill_conv_verify_era5.py). Also saves a
mid-intensification (+48h) node-level snapshot (2401, 3243 codes + MSLP box grid) for the
co-location overlay.

Serialize behind other GPU jobs. Crash-safe per storm: results/skill/convection/run_<name>.npy.

Paper: Sec. 3 'The intervention contrast' (Table tab:mechanism-interventions)
Inputs: GraphCast params (GRAPHCAST_PARAMS); WeatherBench2 ERA5 (GCS, streamed)
Outputs: results/skill/<MECH_RES|convection>/run_<storm>.npy (crash-safe per storm; gain/ramp arms add a suffix)
Run:   # JAX env, GPU (~46 GB)
    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -m graphcast_sae.storms.skill_conv_run
"""
import json, os, sys, time
os.environ["FS_DEVICE"] = "gpu"; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np, jax.numpy as jnp, xarray as xr, dataclasses
import graphcast_sae.common.fs_common as fc
from graphcast import data_utils, rollout
from graphcast_sae.common.signature_physics import gc_km
# SKILL_STORMS selects the storm registry. Default is the frozen TC battery; the
# extratropical battery lives in skill_xt_storms so that appending a storm can
# never silently move the committed TC medians (2026-08-17).
import importlib
S = importlib.import_module("graphcast_sae.common." + os.environ.get("SKILL_STORMS", "skill_conv_storms"))

# MECHANISM OVERRIDE (added 2026-08-13). Same protocol, different feature group, so the
# per-storm bars are comparable. MECH_FEATS must be DISJOINT from the convection group:
# the Ida moisture set shares feature 3174 with convection, which would blend the two.
MECH_NAME = os.environ.get("MECH_NAME", "convection")
_mf = os.environ.get("MECH_FEATS", "")
OUT = fc.ROOT / f"results/skill/{MECH_NAME}"; OUT.mkdir(parents=True, exist_ok=True)
TC = S.TC; RAND = S.RANDOM_CTRL
# MECH_INBOX_CTL points at a per-storm control drawn by inbox_control.py. The frozen
# RANDOM_CTRL is matched on GLOBAL firing rate and fires at a median 2% of the convection
# group's level INSIDE the storm boxes (0% in three of seven) -- a control that ablates
# nothing there cannot fail, which is the control-must-be-able-to-fail rule. The per-storm draw matches on in-box
# firing instead and lands at a median 95%. Absent the env var nothing changes.
INBOX_CTL = json.load(open(os.environ["MECH_INBOX_CTL"])) if os.environ.get(
    "MECH_INBOX_CTL") else {}
# MECH_H extends the rollout past the storm's peak so the DECAY phase is scored too:
# a too-weak storm has little to decay, so if amplification corrects the trajectory
# rather than just the peak, the improvement should survive into dissipation.
H = int(os.environ.get("MECH_H", S.H))
# MECH_GAIN_RELEASE drops the gain to 1.0 (an exact no-op) after this many steps, so
# the model evolves FREELY from a deepened state. Persistence would mean the
# intervention corrected the trajectory; fast relaxation back to baseline would mean
# the weak-storm bias is an attractor the model actively pulls toward -- a stronger
# statement about the defect than the peak error is.
GAIN_RELEASE = int(os.environ.get("MECH_GAIN_RELEASE", "1000000"))
# MECH_RAMPS: LEAD-DEPENDENT gain schedules. The under-deepening is a function of
# forecast lead -- GraphCast captures a median 90% of ERA5's deepening-so-far at
# +18 h decaying monotonically to 50% at +90 h -- so a FLAT gain necessarily
# over-corrects early, where the model is nearly right, and under-corrects late.
# That is why a flat g=2 made Haiyan and Haishen worse: both peak early.
#   lin1to2  linear 1.0 -> 2.0 across the window. Shape assumed, not fitted.
#   invdef   g(lead) = 1 / f(lead), the measured inverse deficit. Shape derived
#            from the deficit curve, which was measured on these same storms, so
#            it is in-sample for SHAPE and must be judged on the held-out four.
# Lead time is known at forecast time, so unlike a per-storm optimum this is a
# schedule a forecaster could actually apply.
_F_LEAD = np.array([6, 18, 30, 42, 54, 66, 78, 90, 96])
_F_CAP = np.array([0.95, 0.90, 0.83, 0.71, 0.59, 0.58, 0.54, 0.50, 0.50])
RAMPS = [x for x in os.environ.get("MECH_RAMPS", "").split(",") if x.strip()]
# MECH_IC_OFFSETS (added 2026-08-20): comma-separated HOURS, e.g. "-48,-24,0,24". Each
# storm is then run once per offset with its INITIAL CONDITION shifted by that many hours,
# and the output file gains a suffix (run_ida2021_m48.npy / _p24.npy) so the runs cannot
# collide. Unset -> [None] -> the storm config is passed through untouched and the file is
# run_<name>.npy exactly as before, so the default path is bit-identical.
#
# WHAT SHIFTS, AND WHAT DELIBERATELY DOES NOT:
#   ic       SHIFTS. It is the only thing the design varies -- the point is N independent
#            draws of the same storm from the same model, differing only in how much of the
#            storm's history the model has already seen.
#   analogs  DOES NOT. They are quiet same-calendar-date dates in OTHER years and they set
#            ftarget, the "normal" level the ablation restores to. Two days of seasonal
#            drift is far below the inter-annual scatter they average over, so shifting them
#            would buy nothing -- but it CAN change which analogs survive the TC>20
#            storm-present screen, which would make ftarget offset-dependent and confound
#            the dose with the offset. Holding them fixed keeps the dose identical across
#            offsets, which is the whole point.
#   box      DOES NOT. It is a fixed tracking box already sized to hold the whole 96 h
#            track; the readout is min MSLP inside it, and a box that moved with the offset
#            would make the readouts non-comparable across offsets.
#   center   DOES NOT. This is the real limitation: `center` fixes the 1500 km ablation
#            disk, and at -48 h the storm sits ~900-1400 km from its +0 h position, i.e.
#            near the disk edge. For BASELINE-ONLY extraction (MECH_ARMS=baseline) the disk
#            is never used, so offsets are exact. For ABLATION arms at |offset| >= 48 h,
#            check the disk exposure before trusting the dose, or add a per-offset centre to
#            the registry. It is not auto-shifted here because inventing a track
#            interpolation would change the dose in a way nothing has audited.
IC_OFFSETS = [int(x) for x in os.environ.get("MECH_IC_OFFSETS", "").split(",") if x.strip()]
_bad = [o for o in IC_OFFSETS if o % 6]
if _bad:   # ERA5/WB2 is 6-hourly; an off-grid IC would die inside load_block's .sel
    raise AssertionError(f"MECH_IC_OFFSETS must be multiples of 6 h; got {_bad}")
if IC_OFFSETS:
    print(f"MECH_IC_OFFSETS: each storm runs at IC offsets {IC_OFFSETS} h "
          "(analogs, box and disk centre are held FIXED -- see the note in the source)",
          flush=True)

def _off_tag(off):
    """Filename suffix for an IC offset. None -> '' (today's filename, unchanged)."""
    if off is None:
        return ""
    return f"_m{-int(off)}" if off < 0 else f"_p{int(off)}"

def _shift_ic(ic, off):
    """Shift a storm's IC date string by `off` hours, keeping it parseable by load_block."""
    t = np.datetime64(ic) + np.timedelta64(int(off), "h")
    return str(np.datetime64(t, "h"))

def _sched(kind, n):
    leads = 6.0 * (1 + np.arange(n))
    if kind == "lin1to2":
        return np.linspace(1.0, 2.0, n).astype(np.float32)
    if kind == "invdef":
        return (1.0 / np.interp(leads, _F_LEAD, _F_CAP)).astype(np.float32)
    # pulse<k> (added 2026-08-20): dose ONE step and leave every other step untouched.
    # A flat arm confounds "the feature matters" with "16 consecutive perturbations
    # compound", and a ramp still doses every step. The pulse isolates a single
    # intervention time, which is what a lag-resolved transmission measurement needs:
    # the response at step k+tau to a dose at step k, with nothing else applied.
    # g=0 at step k IS the committed restore-to-normal arm (delta_gain's gain=0 is
    # exactly delta_cond); g=1 everywhere else is an EXACT no-op -- delta_gain computes
    # (1-1)*excess = 0, so those steps are bit-identical to baseline.
    if kind.startswith("pulse"):
        k = int(kind[5:])
        if not 0 <= k < n:
            raise ValueError(f"{kind}: step {k} outside the rollout (H={n}, steps 0..{n-1})")
        g = np.ones(n, np.float32)
        g[k] = 0.0
        return g
    raise ValueError(f"unknown ramp {kind}")
CONV = [int(x) for x in _mf.split(",")] if _mf else S.CONV
# OUTCOME DISJOINTNESS (added 2026-08-17, after mech_atm_river was voided). The overlap
# check below guards against blending two TREATMENT groups; nothing guarded against
# ablating the READOUT. mech_atm_river ran with MECH_FEATS=3243,... and 3243 is TC, so it
# deleted the feature the verdict scores and returned the largest effect in the library
# by 3x. That failure mode is self-camouflaging -- it presents as the best arm -- so it
# gets a hard assert with no override.
for _grp, _lbl in ((CONV, "ablation group"), (RAND, "random control")):
    if TC in _grp:
        raise AssertionError(
            f"{MECH_NAME}: {_lbl} {_grp} contains the OUTCOME feature TC={TC}. Ablating the "
            "readout is circular -- it suppresses the metric by construction. Drop it.")
if MECH_NAME != "convection":
    _ov = set(CONV) & set(S.CONV)
    if _ov and os.environ.get("MECH_ALLOW_OVERLAP") != "1":
        raise AssertionError(f"{MECH_NAME} group overlaps convection {S.CONV} at {sorted(_ov)}; "
                             "set MECH_ALLOW_OVERLAP=1 only if the overlap is INTENDED "
                             "(e.g. 'ascent' IS the convection mechanism under its purified name)")
    if _ov:
        print(f"  NOTE: {MECH_NAME} shares {sorted(_ov)} with the convection group -- intended, "
              "this is the same mechanism with purified features", flush=True)
    print(f"MECHANISM OVERRIDE: {MECH_NAME} features {CONV} (convection was {S.CONV})", flush=True)
MID = 7  # +48h snapshot for co-location
# MECH_TRACK: track every concept's features, not just the dosed group, so ONE run
# yields both readouts -- physical necessity AND the concept-concept interaction matrix,
# measured inside the event where the concepts are actually active.
# MECH_TRACK=all tracks every feature, which is how the exposure probe measures
# in-box activation for all candidate mechanism groups in ONE baseline rollout
# instead of one GPU run per group.
_tv = os.environ.get('MECH_TRACK', '')
if _tv.strip().lower() == 'all':
    _trk = list(range(4096))
    print("MECH_TRACK=all: tracking every feature (exposure probe)", flush=True)
else:
    _trk = [int(x) for x in _tv.split(',') if x.strip()]
FEAT_TRACK = sorted(set([TC] + CONV + RAND + _trk
                        + [f for v in INBOX_CTL.values() for f in v["rand"]]))
GAINS = [float(x) for x in os.environ.get("MECH_GAINS", "").split(",") if x.strip()]
# MECH_FIELDS: also store the SPATIAL readout per arm per lead -- the MSLP and 10 m wind
# grids over the storm box, the located centre (for track error), and the cyclone and
# convection features on the mesh nodes inside the box. Off by default, because it adds
# ~10 MB per storm and every committed run was scored without it.
FIELDS = os.environ.get("MECH_FIELDS", "") == "1"
if FIELDS:
    print("MECH_FIELDS=1: storing MSLP/wind grids, storm centres and in-box node features "
          "for every arm and lead", flush=True)
if GAINS:
    print(f"GAIN SWEEP: g = {GAINS}  (g=0 == the committed conv-normal arm, "
          f"g=1 == baseline, g>1 amplifies)", flush=True)

def numpyify(ds):
    return xr.Dataset({v: (ds[v].dims, np.asarray(ds[v].values)) for v in ds.data_vars},
                      coords={k: ds.coords[k] for k in ds.coords})

def build_io(t0, H, task_config):
    blk, times, st = fc.load_block(np.datetime64(t0), nframes=2 + H)
    w = blk.assign_coords(time=(times - times[0]).astype("timedelta64[ns]"))
    for v in list(w.data_vars):
        if "time" in w[v].dims: w[v] = w[v].expand_dims("batch")
    for v in fc.STATIC_VARS: w[v] = st[v]
    w = w.assign_coords(datetime=(("batch", "time"), times[None, :].astype("datetime64[ns]")))
    return data_utils.extract_inputs_targets_forcings(
        w, target_lead_times=slice("6h", f"{6*H}h"), **dataclasses.asdict(task_config))

def _box_sel(p, box):
    """The three surface fields cropped to the storm box, longitude wrap handled."""
    la0, la1 = box["lat"]; lo = S.norm_lon(box["lon"])
    m = p["mean_sea_level_pressure"].sel(lat=slice(la0, la1))
    u = p["10m_u_component_of_wind"].sel(lat=slice(la0, la1))
    v = p["10m_v_component_of_wind"].sel(lat=slice(la0, la1))
    if lo[0] <= lo[1]:
        sl = dict(lon=slice(lo[0], lo[1])); m = m.sel(**sl); u = u.sel(**sl); v = v.sel(**sl)
    else:
        mask = (p.lon >= lo[0]) | (p.lon <= lo[1])
        m = m.sel(lon=mask); u = u.sel(lon=mask); v = v.sel(lon=mask)
    return m, u, v

def box_phys(p, box):
    """min MSLP (hPa) & max 10m wind (m/s) in the box from a numpyified single-time pred."""
    m, u, v = _box_sel(p, box)
    mslp_min = float(np.nanmin(m.values)) / 100.0
    wind_max = float(np.nanmax(np.sqrt(u.values ** 2 + v.values ** 2)))
    return mslp_min, wind_max

def box_fields(p, box):
    """MECH_FIELDS: the SPATIAL readout, per lead per arm.

    The scalar readout (min MSLP in a box) cannot distinguish a storm that is too weak from
    one that is in the wrong place, and it cannot show what an intervention does to
    propagation. This returns the field itself plus the located centre, so track error and
    intensity error can be scored separately -- the standard TC verification split, and the
    one on which data-driven models are known to behave very differently.

    Centre = argmin MSLP over the box. That is the same estimator the scalar readout uses,
    so the two are consistent by construction; it is crude for a weak or double-centred
    system, which is why the centre is stored alongside the field rather than instead of it.
    """
    m, u, v = _box_sel(p, box)
    mg = np.asarray(m.values, np.float32).squeeze() / 100.0          # hPa
    wg = np.asarray(np.sqrt(u.values ** 2 + v.values ** 2), np.float32).squeeze()
    lat = np.asarray(m.lat.values, np.float32); lon = np.asarray(m.lon.values, np.float32)
    j, i = np.unravel_index(int(np.nanargmin(mg)), mg.shape)
    return dict(mslp=mg, wind=wg, clat=float(lat[j]), clon=float(lon[i])), lat, lon

def run_storm(name, cfg, sae, apply, mlat, mlon, params_ok=True, suffix="", ic_offset=None):
    center = cfg["center"]; box = cfg["box"]
    # per-storm control shadows the module-level RAND for the whole of this function
    RAND = INBOX_CTL[name]["rand"] if name in INBOX_CTL else S.RANDOM_CTRL
    if name in INBOX_CTL:
        print(f"[{name}] IN-BOX matched control {RAND} "
              f"(in-box {sum(INBOX_CTL[name]['new_inbox']):.1f} vs convection "
              f"{sum(INBOX_CTL[name]['conv_inbox']):.1f}; the global-rate control fired "
              f"{sum(INBOX_CTL[name]['old_inbox']):.1f})", flush=True)
    nmask = (gc_km(mlat, mlon, center[0], center[1]) < S.RADIUS_KM).astype(np.float32)
    inbox = (mlat >= box["lat"][0]) & (mlat <= box["lat"][1]) & (mlon >= box["lon"][0]) & (mlon <= box["lon"][1])
    print(f"[{name}] disk={int(nmask.sum())} nodes; box={int(inbox.sum())} mesh nodes", flush=True)

    def codes_at(t0):
        blk = fc.load_block(np.datetime64(t0), nframes=fc.INPUT_WINDOW)
        inp, tg, fr = fc.build_batch_inputs([blk], 0, tc)
        z = jnp.zeros(sae.n_features, jnp.float32)
        _, acts = apply(inp, tg, fr, (z, z, np.zeros(len(mlat), np.float32)))
        X = np.asarray(acts, np.float32).reshape(-1, fc.D_IN)
        return np.asarray(sae.codes(jnp.asarray(X)))

    # ---- NORMAL reference from quiet, no-storm analogs (for CONV and RANDOM) ----
    # MECH_FTARGET=<json>: per-storm frozen normal levels {storm: {feat: level}}, taken
    # from a previous battery instead of re-estimated on this run's disk. Added 2026-08-23
    # (prereg_conv_radius_sweep amendment): estimating the reference on the disk itself
    # makes restore-to-normal degenerate to delete on small disks (at 750 km, Ida and
    # Haiyan have all three convection targets at 0), so an extent sweep must freeze the
    # reference and vary only the mask.
    frozen = os.environ.get("MECH_FTARGET")
    accf = {f: [] for f in CONV + RAND}
    used = []
    for a in ([] if frozen else cfg["analogs"]):
        try:
            c = codes_at(a)
        except Exception as e:
            print(f"  analog {a}: load ERROR {e}", flush=True); continue
        storm = c[inbox, TC].sum()
        if storm > 20:
            print(f"  analog {a}: TC={storm:.0f} storm present, SKIP", flush=True); continue
        for f in CONV + RAND:
            v = c[nmask.astype(bool), f]; accf[f].extend(v[v > 0].tolist())
        used.append(a); print(f"  analog {a}: TC={storm:.0f} quiet, used", flush=True)
    ftarget = np.zeros(sae.n_features, np.float32)
    if frozen:
        froz = {int(k): float(v) for k, v in json.load(open(frozen))[name].items()}
        missing = [f for f in CONV + RAND if f not in froz]
        assert not missing, f"MECH_FTARGET lacks features {missing} for {name}"
        for f in CONV + RAND:
            ftarget[f] = froz[f]
        used = ["frozen:" + frozen]
        print(f"  FROZEN normal levels CONV {[round(float(ftarget[f]),2) for f in CONV]} "
              f"RAND {[round(float(ftarget[f]),2) for f in RAND]} (from {frozen})", flush=True)
    else:
        for f in CONV + RAND:
            ftarget[f] = np.mean(accf[f]) if accf[f] else 0.0
        print(f"  normal levels CONV {[round(float(ftarget[f]),2) for f in CONV]} "
              f"RAND {[round(float(ftarget[f]),2) for f in RAND]}", flush=True)

    # ---- arms ----
    fsel_conv = np.zeros(sae.n_features, np.float32); fsel_conv[CONV] = 1.0
    fsel_rand = np.zeros(sae.n_features, np.float32); fsel_rand[RAND] = 1.0
    zeroF = np.zeros(sae.n_features, np.float32); zeroN = np.zeros(len(mlat), np.float32)
    arms = {
        "baseline":    (zeroF, zeroF, zeroN),
        "conv-normal": (fsel_conv, ftarget, nmask),
        "conv-zero":   (fsel_conv, zeroF, nmask),
    }
    # MECH_GAINS: sweep the excess-above-normal scaling instead of the fixed arms.
    # g=0 IS conv-normal (verified bit-identical in fs_common.delta_gain), g=1 is
    # baseline, g>1 amplifies -- so this asks the other half of the causal question
    # (what if there were MORE convection) on the same instrument. Both ends of the
    # curve are reported; a single amplified gain on its own is not interpretable.
    if GAINS or RAMPS:
        arms = {"baseline": (zeroF, zeroF, zeroN)}
        for g in GAINS:
            arms[f"gain-{g:g}"] = (fsel_conv, ftarget, nmask, np.float32(g))
        for kind in RAMPS:
            arms[f"ramp-{kind}"] = (fsel_conv, ftarget, nmask, _sched(kind, H))
    # An empty RAND means no control has been drawn for this battery yet (the
    # exposure probe). Skip the arm rather than run a no-op and report it as a
    # passing control -- a control that ablates nothing cannot fail.
    if RAND:
        arms["rand-normal"] = (fsel_rand, ftarget, nmask)
        if GAINS:   # matched-gain control: amplify the RANDOM group by the largest g
            arms[f"rand-gain-{max(GAINS):g}"] = (fsel_rand, ftarget, nmask,
                                                 np.float32(max(GAINS)))
        # matched-SCHEDULE control (added 2026-08-20). The rand-gain arm above is gated on
        # GAINS, so before this a MECH_RAMPS run produced convection ramp/pulse arms and NO
        # control arm at all -- a schedule sweep with nothing to fail against certifies
        # nothing (the control-must-be-able-to-fail rule). The control gets the IDENTICAL schedule, so a pulse at
        # step k is compared against the same group-size, same-dose, same-timing ablation of
        # a group that is not the mechanism. Note this DOUBLES the rollouts in a ramp sweep.
        for kind in RAMPS:
            arms[f"rand-ramp-{kind}"] = (fsel_rand, ftarget, nmask, _sched(kind, H))
    else:
        print("  NO random control drawn for this battery -- rand-normal arm SKIPPED "
              "(this run cannot be scored as a controlled result)", flush=True)

    # MECH_ARMS restricts which arms run, so a snapshot-only pass (MECH_ARMS=baseline)
    # can refresh the stored +48 h node activations without repeating every ablation.
    _keep = [x for x in os.environ.get("MECH_ARMS", "").split(",") if x.strip()]
    if _keep:
        missing = [k for k in _keep if k not in arms]
        if missing:
            raise AssertionError(f"MECH_ARMS names arms that do not exist: {missing}; "
                                 f"available {sorted(arms)}")
        arms = {k: arms[k] for k in _keep}
        print(f"  MECH_ARMS restricts this run to {list(arms)}", flush=True)

    inp, tgt, frc = build_io(cfg["ic"], H, tc)
    tct = tgt.time.isel(time=slice(0, 1))
    for cc in ("datetime",):
        if cc in tgt.coords: tgt = tgt.drop_vars(cc)
        if cc in frc.coords: frc = frc.drop_vars(cc)

    def boxfeats(a):
        X = jnp.asarray(np.asarray(a, np.float32).reshape(-1, fc.D_IN))
        C = np.asarray(sae.codes(X))
        return {f: float(C[inbox, f].sum()) for f in FEAT_TRACK}, C

    def roll(patch, capture_mid=False):
        cur = inp; per = {f: [] for f in FEAT_TRACK}
        mslp = []; wind = []; snap = None
        fld = {"mslp": [], "wind": [], "clat": [], "clon": [], "nodefeat": []}
        glat = glon = None
        pj0 = tuple(jnp.asarray(x) for x in patch)
        # MECH_RAMPS hands the gain in as a LENGTH-H SCHEDULE, but the patch tuple was built
        # once outside the loop and delta_gain does `(gain - 1) * excess` with excess of
        # shape (n_nodes, n_features) -- so a (16,) gain raised
        #     broadcast_shapes got incompatible shapes: (1, 16), (40962, 4096)
        # and every ramp arm ever launched died on its first patched step. A lead-dependent
        # gain has to be INDEXED per step, which is the whole point of it being lead-dependent.
        ramped = len(pj0) == 4 and jnp.ndim(pj0[3]) == 1
        for h in range(H):
            pj = pj0[:3] + (pj0[3][h],) if ramped else pj0
            ct = tgt.isel(time=slice(h, h+1)).assign_coords(time=tct)
            cf = frc.isel(time=slice(h, h+1)).assign_coords(time=tct)
            p, a = apply(cur, ct, cf, pj)
            bf, C = boxfeats(a)
            for f in FEAT_TRACK: per[f].append(bf[f])
            p = numpyify(p)
            mm, ww = box_phys(p, box); mslp.append(mm); wind.append(ww)
            if FIELDS:
                d, glat, glon = box_fields(p, box)
                fld["mslp"].append(d["mslp"]); fld["wind"].append(d["wind"])
                fld["clat"].append(d["clat"]); fld["clon"].append(d["clon"])
                fld["nodefeat"].append(C[np.ix_(inbox, [TC] + CONV)].astype(np.float32))
            if capture_mid and h == MID:
                la0, la1 = box["lat"]; lo = S.norm_lon(box["lon"])
                mg = p["mean_sea_level_pressure"].sel(lat=slice(la0, la1))
                mg = mg.sel(lon=slice(lo[0], lo[1])) if lo[0] <= lo[1] else mg
                # node_rand added 2026-08-18 for PS-5's guardrail-#9 control. The snapshot
                # is taken on the UNPATCHED baseline arm, so these are the same forward
                # pass -- storing three more columns costs nothing and is the only way to
                # ask whether the shear-displacement statistic reads the convection
                # features specifically or just the storm's own asymmetry.
                snap = dict(node_2401=C[:, 2401].astype(np.float32), node_3243=C[:, TC].astype(np.float32),
                            node_conv=C[:, CONV].astype(np.float32),
                            node_rand=(C[:, RAND].astype(np.float32) if RAND else None),
                            mlat=mlat.astype(np.float32),
                            mlon=mlon.astype(np.float32),
                            mslp_grid=np.asarray(mg.values, np.float32).squeeze(),
                            mslp_lat=np.asarray(mg.lat.values), mslp_lon=np.asarray(mg.lon.values))
            if h < H-1:
                cur = rollout._get_next_inputs(cur, xr.merge([p, cf])).assign_coords(time=cur.coords["time"])
        out = dict(box_feats={f: np.array(per[f]) for f in FEAT_TRACK},
                   mslp_min=np.array(mslp), wind_max=np.array(wind))
        if FIELDS:
            out["fields"] = dict(
                mslp=np.stack(fld["mslp"]).astype(np.float32),
                wind=np.stack(fld["wind"]).astype(np.float32),
                clat=np.array(fld["clat"], np.float32), clon=np.array(fld["clon"], np.float32),
                nodefeat=np.stack(fld["nodefeat"]).astype(np.float32),
                grid_lat=glat, grid_lon=glon)
        return out, snap

    res = {}; snaps = {}
    for aname, patch in arms.items():
        t = time.time()
        r, snap = roll(patch, capture_mid=(aname == "baseline"))
        res[aname] = r
        if snap is not None: snaps["baseline_mid"] = snap
        print(f"  [{aname}] MSLP min {np.array2string(r['mslp_min'],precision=1,max_line_width=250)}", flush=True)
        print(f"  [{aname}] TC3243  {np.array2string(r['box_feats'][TC],precision=0,max_line_width=250)}  ({time.time()-t:.0f}s)", flush=True)

    out = dict(name=name, ic=cfg["ic"], center=center, box=box, basin=cfg.get("basin"),
               nondev=bool(cfg.get("nondev", False)), analogs_used=used,
               ftarget={f: float(ftarget[f]) for f in CONV + RAND}, conv=CONV, rand=RAND, tc=TC,
               mech=MECH_NAME,
               res=res, snap=snaps, disk_nodes=int(nmask.sum()), box_nodes=int(inbox.sum()))
    if FIELDS:   # node coordinates for the in-box feature stack, and the arm's own gains
        out["box_node_lat"] = mlat[inbox].astype(np.float32)
        out["box_node_lon"] = mlon[inbox].astype(np.float32)
        out["nodefeat_ids"] = [TC] + list(CONV)
    # only present when MECH_IC_OFFSETS is in use, so the default payload is unchanged
    if ic_offset is not None:
        out["ic_offset_h"] = int(ic_offset)
    np.save(OUT / f"run_{name}{suffix}.npy", out, allow_pickle=True)
    print(f"  -> run_{name}{suffix}.npy", flush=True)
    return out

def main():
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    geom = np.load(fc.MESH_GEOM, allow_pickle=True).item()
    mlat = geom["lat"]; mlon = np.where(geom["lon"] > 180, geom["lon"] - 360, geom["lon"])
    global tc
    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply_cond(mc, tc, stats, sae, bf16=True)
    apply = fc.make_apply(params, rf, patched=True)
    print(f"model+SAE loaded; features={sae.n_features}; storms={list(S.STORMS)}", flush=True)
    for name, cfg in S.STORMS.items():
        if only and name not in only: continue
        # IC_OFFSETS empty -> [None] -> one pass, unshifted cfg, suffix '': today's behaviour
        for off in (IC_OFFSETS or [None]):
            cfg_o = cfg if off is None else dict(cfg, ic=_shift_ic(cfg["ic"], off))
            sfx = _off_tag(off)
            fpath = OUT / f"run_{name}{sfx}.npy"
            if fpath.exists():
                print(f"[{name}{sfx}] exists, skip", flush=True); continue
            if off is not None:
                print(f"[{name}{sfx}] IC {cfg['ic']} {off:+d} h -> {cfg_o['ic']}", flush=True)
            t = time.time()
            run_storm(name, cfg_o, sae, apply, mlat, mlon, suffix=sfx, ic_offset=off)
            print(f"[{name}{sfx}] done in {(time.time()-t)/60:.1f} min\n", flush=True)

if __name__ == "__main__":
    main()
