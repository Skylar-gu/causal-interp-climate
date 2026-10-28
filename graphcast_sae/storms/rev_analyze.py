"""Score the camera-ready revision runs (docs/prereg/prereg_cluster_revision.md, frozen before any number).

    MECH_RES=suff_ctl       python -m graphcast_sae.storms.rev_analyze j1     # J1 + J3 (+ gates 2-3)
    MECH_RES=determinism_j4 python -m graphcast_sae.storms.rev_analyze j4     # J4 per-arm floor
    SKILL_STORMS=skill_sh_storms MECH_RES=sh_suff_ctl python -m graphcast_sae.storms.rev_analyze j1

Per-arm metrics use skill_conv_analyze.metrics' exact definitions (deepening = IC MSLP - min
forecast MSLP; err = RMSE of box min-MSLP vs ERA5 over leads up to max(ERA5 peak, +36 h)),
re-implemented here only because that function hard-codes its four arm names.
Writes results/skill/<MECH_RES>/rev_verdict.json. Never differences arms across runs.

Paper: Table tab:ledger, Fig. 2, Sec. 3.3 (J1 gain sweep, J3 readout, J4 floor, J7 Southern Hemisphere)
Inputs: results/skill/<MECH_RES>/run_*.npy (not shipped, written by storms/skill_conv_run.py)
Outputs: results/skill/<MECH_RES>/rev_verdict.json
Run:   # JAX env, CPU
    MECH_RES=suff_ctl python -m graphcast_sae.storms.rev_analyze j1
"""
import importlib, itertools, json, os, sys
import numpy as np
import graphcast_sae.common.fs_common as fc

S = importlib.import_module("graphcast_sae.common." + os.environ.get("SKILL_STORMS", "skill_conv_storms"))
RES = fc.ROOT / f"results/skill/{os.environ['MECH_RES']}"
TC = S.TC
MID = 7                       # +48 h, the Fig 3 lead
FLOOR_INFERRED = 0.37         # p90 of the inferred floor; used until J4 exists
IDA_CONV_NORMAL = 2.675546875000009   # convection_inboxctl verdict, Ida conv-normal d_deepen


def arm_metrics(r, t):
    ic = float(t["mslp_min"][0]); era_peak = int(np.argmin(t["mslp_min"]))
    era_dp = ic - float(np.min(t["mslp_min"]))
    res = r["res"]; H = len(res["baseline"]["mslp_min"])
    truth = t["mslp_min"][1:H + 1]; w = np.arange(min(max(era_peak, 6), H))
    out = {}
    for a, x in res.items():
        mm = x["mslp_min"]
        d = dict(deepen=ic - float(np.min(mm)),
                 err_mslp=float(np.sqrt(np.mean((mm[w] - truth[w]) ** 2))),
                 # the PRE-REGISTERED bar metric (prereg_cluster_revision.md "deepening-error"),
                 # = result_sufficiency.md's |err|: |ERA5 deepening - forecast deepening|
                 dep_err=abs(era_dp - (ic - float(np.min(mm)))),
                 tc_mid_post=float(x["box_feats"][TC][MID]),
                 tc_peak_post=float(np.max(x["box_feats"][TC])))
        if "box_feats_pre" in x:
            pre = x["box_feats_pre"][TC]
            # pre-patch capture at step k reads the state the patch at steps < k produced; the
            # "t+1" readout of a +48 h dose is therefore index MID+1 (+54 h); both are stored
            d.update(tc_mid_pre=float(pre[MID]), tc_mid1_pre=float(pre[min(MID + 1, H - 1)]),
                     tc_peak_pre=float(np.max(pre)))
        if "fields" in x and "collat" in x["fields"]:
            c = x["fields"]["collat"]
            d["collat_96h"] = {k: float(v[-1]) for k, v in c.items()}
            d["disk_q999_max"] = [float(v) for v in x["fields"]["disk_q"][:, 1, :].max(0)]
            d["disk_max"] = [float(v) for v in x["fields"]["disk_q"][:, 2, :].max(0)]
        out[a] = d
    b = out["baseline"]
    for a, d in out.items():
        d["d_deepen"] = b["deepen"] - d["deepen"]          # >0 => less deepening than baseline
        d["err_red"] = b["dep_err"] - d["dep_err"]         # BAR metric: >0 => deepening closer to ERA5
        d["rmse_red"] = b["err_mslp"] - d["err_mslp"]      # secondary: min-MSLP trajectory RMSE
        if "collat_96h" in d:
            d["d_collat_96h"] = {k: d["collat_96h"][k] - b["collat_96h"][k] for k in d["collat_96h"]}
    return dict(ic_mslp=ic, era_deepen=ic - float(np.min(t["mslp_min"])),
                nondev=bool(r.get("nondev", False)), arms=out,
                provenance=r.get("provenance"))


def load():
    truth = np.load(RES / "era5_truth.npy", allow_pickle=True).item()
    runs = {n: np.load(RES / f"run_{n}.npy", allow_pickle=True).item()
            for n in S.STORMS if (RES / f"run_{n}.npy").exists()}
    return truth, runs


def j4_floor():
    p = fc.ROOT / "results/skill/determinism_j4/rev_verdict.json"
    if p.exists():
        return json.load(open(p))["floor"]["d_deepen_max"], "J4 measured"
    return FLOOR_INFERRED, "inferred p90 (PROVISIONAL: J4 not yet landed)"


def j1(truth, runs):
    M = {n: arm_metrics(r, truth[n]) for n, r in runs.items()}
    floor, fsrc = j4_floor()
    gates = {}
    for n, r in runs.items():
        res = r["res"]
        if "gain-1" in res:
            gates.setdefault("g1_bit_identical", {})[n] = bool(
                np.array_equal(res["gain-1"]["mslp_min"], res["baseline"]["mslp_min"]))
    if "ida2021" in M and "gain-0" in M["ida2021"]["arms"] and S.__name__ == "skill_conv_storms":
        dd = M["ida2021"]["arms"]["gain-0"]["d_deepen"]
        gates["ida_conv_normal_repro"] = dict(d_deepen=dd, committed=IDA_CONV_NORMAL,
                                              diff=dd - IDA_CONV_NORMAL,
                                              passed=bool(abs(dd - IDA_CONV_NORMAL) <= FLOOR_INFERRED))
    gains = sorted({float(a[5:]) for m in M.values() for a in m["arms"] if a.startswith("gain-")})
    table = {}
    for n, m in M.items():
        row = {}
        for g in gains:
            t, c = m["arms"].get(f"gain-{g:g}"), m["arms"].get(f"rand-gain-{g:g}")
            if t is None or c is None:
                continue
            row[f"{g:g}"] = dict(err_red_treat=t["err_red"], err_red_ctl=c["err_red"],
                                 spec=t["err_red"] - c["err_red"],
                                 spec_rmse=t["rmse_red"] - c["rmse_red"],
                                 d_deepen_treat=t["d_deepen"], d_deepen_ctl=c["d_deepen"])
        table[n] = row
    dev = [n for n in M if not M[n]["nondev"]]
    g2 = {n: table[n].get("2", {}).get("spec") for n in dev}
    passes = [n for n, v in g2.items() if v is not None and v > floor]
    need = 5 if S.__name__ == "skill_conv_storms" else 4
    # guardrail #9: does the control move anything beyond the floor at any gain, anywhere?
    ctl_moves = {n: max((abs(v["d_deepen_ctl"]) for v in table[n].values()), default=0.0) for n in M}
    j3 = {}
    for n, m in M.items():
        a = m["arms"]
        if "gain-2" in a and "tc_mid_pre" in a["gain-2"]:
            b = a["baseline"]
            j3[n] = dict(base_post=b["tc_mid_post"], base_pre=b["tc_mid_pre"],
                         g2_post=a["gain-2"]["tc_mid_post"], g2_pre=a["gain-2"]["tc_mid_pre"],
                         g2_pre_t1=a["gain-2"]["tc_mid1_pre"],
                         ctl2_pre_t1=a.get("rand-gain-2", {}).get("tc_mid1_pre"),
                         base_pre_t1=b["tc_mid1_pre"],
                         leak=a["gain-2"]["tc_mid_post"] - a["gain-2"]["tc_mid_pre"])
    nondev = {n: {f"{g:g}": M[n]["arms"][f"gain-{g:g}"]["d_deepen"] for g in gains
                  if g <= 2 and f"gain-{g:g}" in M[n]["arms"]} for n in M if M[n]["nondev"]}
    verdict = dict(
        floor=floor, floor_source=fsrc, gates=gates, spec_table=table,
        bar=dict(rule=f"spec(g=2) = err_red(treat) - err_red(in-box ctl) > floor in >= {need} of {len(dev)} developing",
                 g2_spec=g2, passing=passes, n_pass=len(passes), need=need,
                 claim_specific_sufficiency=len(passes) >= need and max(ctl_moves.values(), default=0) > floor),
        guardrail9_ctl_max_abs_d_deepen=ctl_moves, j3_tc3243_48h=j3,
        false_alarm_nondev_neg_d_deepen=nondev,   # negative d_deepen = deepened more than baseline
        metrics=M)
    return verdict


def j4(truth, runs):
    out = {}; worst_dd = worst_traj = 0.0
    for n, r in runs.items():
        M = arm_metrics(r, truth[n]); res = r["res"]; per = {}
        for base in ("baseline", "gain-0", "conv-normal"):
            names = [a for a in res if a == base or a.startswith(base + "#r")]
            if len(names) < 2:
                continue
            dd = [M["arms"][a]["deepen"] for a in names]
            traj = max(float(np.max(np.abs(res[a]["mslp_min"] - res[b]["mslp_min"])))
                       for a, b in itertools.combinations(names, 2))
            per[base] = dict(n=len(names), deepen=dd, deepen_range=float(np.ptp(dd)),
                             max_pairwise_traj_hpa=traj,
                             bit_identical=all(np.array_equal(res[names[0]]["mslp_min"], res[a]["mslp_min"])
                                               for a in names[1:]))
            worst_dd = max(worst_dd, float(np.ptp(dd))); worst_traj = max(worst_traj, traj)
        out[n] = per
    # d_deepen of conv-normal vs baseline mixes two arms' noise: floor on a difference = sum of ranges
    rng = [v["deepen_range"] for p in out.values() for v in p.values()]
    return dict(per_storm=out,
                floor=dict(deepen_range_max=worst_dd, d_deepen_max=2 * worst_dd if rng else None,
                           traj_max_hpa=worst_traj,
                           note="d_deepen floor = 2 x worst single-arm range (difference of two noisy arms)"))


def boot_ci(x, n=10000, seed=0):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)])
    if len(x) == 0:
        return dict(median=None, lo=None, hi=None, n=0)
    rng = np.random.default_rng(seed)
    meds = np.median(x[rng.integers(0, len(x), (n, len(x)))], axis=1)
    return dict(median=float(np.median(x)), lo=float(np.percentile(meds, 2.5)),
                hi=float(np.percentile(meds, 97.5)), n=int(len(x)))


def j2(truth, runs):
    """Prereg amendment 2: SAE direction vs random (B1) and difference-of-means (B3)."""
    M = {n: arm_metrics(r, truth[n]) for n, r in runs.items()}
    gains = sorted({float(a[5:]) for m in M.values() for a in m["arms"] if a.startswith("gain-")})
    per = {}
    for n, m in M.items():
        a = m["arms"]; row = {}
        for g in gains:
            t = a.get(f"gain-{g:g}", {}).get("err_red")
            rr = [a[k]["err_red"] for k in a if k.startswith("dir-rand") and k.endswith(f"-{g:g}")]
            dm = a.get(f"dir-dom-{g:g}", {}).get("err_red")
            row[f"{g:g}"] = dict(treat=t, rand=rr, rand_med=float(np.median(rr)) if rr else None, dom=dm,
                                 t_minus_rand=(t - float(np.median(rr))) if rr and t is not None else None,
                                 t_minus_dom=(t - dm) if dm is not None and t is not None else None)
        best = {}   # each arm family at its OWN best gain (in-sample, same rule for all)
        for fam in ("treat", "rand_med", "dom"):
            vals = [(row[k][fam], k) for k in row if row[k][fam] is not None]
            best[fam] = max(vals) if vals else None
        per[n] = dict(by_gain=row, best_gain=best,
                      cos_dom_conv=runs[n].get("j2_dirs", {}).get("cos_dom_conv"))
    dev = [n for n in M if not M[n]["nondev"]]
    summ = {g: dict(t_minus_rand=boot_ci([per[n]["by_gain"][g]["t_minus_rand"] for n in dev]),
                    t_minus_dom=boot_ci([per[n]["by_gain"][g]["t_minus_dom"] for n in dev]))
            for g in (f"{x:g}" for x in gains)}
    s2 = summ.get("2", {}).get("t_minus_rand", {})
    return dict(per_storm=per, summary_dev=summ,
                direction_specific=bool(s2.get("lo") is not None and s2["lo"] > 0),
                rule="direction-specific iff treat - median(B1) at g=2 has bootstrap CI > 0",
                metrics=M)


def main():
    mode = sys.argv[1]
    truth, runs = load()
    print(f"{RES.name}: {len(runs)} storms on disk: {list(runs)}")
    v = {"j1": j1, "j2": j2, "j4": j4}[mode](truth, runs)
    p = RES / "rev_verdict.json"
    json.dump(v, open(p, "w"), indent=1, default=float)
    print(json.dumps({k: v[k] for k in v if k != "metrics"}, indent=1, default=float)[:6000])
    print("->", p)


if __name__ == "__main__":
    main()
