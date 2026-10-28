"""Stage a SPARSE local copy of the WB2 0.25°/37-lev ERA5 zarr for clusters whose compute
nodes have no internet (Rorqual, 2026-10-12).

Copies raw chunk objects only -- no decode, no re-encode -- so the local store is the remote
store byte-for-byte at the requested timestamps. Every time-dependent variable is chunked
(1, ...) along time, so one timestamp = one object per variable. Chunks never requested are
simply absent; zarr fills them with the fill value (NaN), and fs_common.load_block asserts
finiteness on what it loads, so a missing frame fails loudly instead of silently.

    python -m graphcast_sae.extraction.stage_era5_local --out $GC_SCRATCH/era5_wb2.zarr --jobs conv,sh,gridlock
Run on a node WITH internet (login/DTN). Network-bound; stdlib only.

Read by fs_common.load_block when FS_ERA5_LOCAL points at the output.
Paper: infrastructure: ERA5 staging for clusters without internet
Inputs: WeatherBench2 ERA5 (public HTTPS)
Outputs: a sparse zarr at --out
"""
import argparse, concurrent.futures as cf, json, os, pathlib, sys, urllib.request
import numpy as np

BASE = "https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2022-full_37-6h-0p25deg_derived.zarr"
T0 = np.datetime64("1959-01-01T00")
TIME_VARS = ("2m_temperature", "mean_sea_level_pressure", "10m_v_component_of_wind",
             "10m_u_component_of_wind", "total_precipitation_6hr", "temperature", "geopotential",
             "u_component_of_wind", "v_component_of_wind", "vertical_velocity",
             "specific_humidity", "toa_incident_solar_radiation")
WHOLE = ("geopotential_at_surface", "land_sea_mask", "latitude", "longitude", "level", "time")
STEP = np.timedelta64(6, "h")


def frames(center, n):
    """fs_common.load_block semantics: n frames starting at center-6h."""
    return [np.datetime64(center, "h") - STEP + k * STEP for k in range(n)]


ONLY = None   # --storms filter


def storm_times(mod, H=16):
    out = []
    for name, cfg in mod.STORMS.items():
        if ONLY and name not in ONLY:
            continue
        out += frames(cfg["ic"], 2 + H)
        for a in cfg["analogs"]:
            out += frames(a, 3)
    return out


def wanted(jobs):
    ts = []
    if "conv" in jobs:
        import graphcast_sae.common.skill_conv_storms as S; ts += storm_times(S)
    if "sh" in jobs:
        import graphcast_sae.common.skill_sh_storms as S; ts += storm_times(S)
    if "gridlock" in jobs:   # scripts/cluster/ablate_six.sh: 6 ICs, GR_S=8 -> IC-6h .. IC+48h
        for ic in ("2020-01-10", "2020-03-12", "2020-05-11", "2020-07-10", "2020-09-08", "2020-11-07"):
            ts += frames(ic, 2 + 8)
    return sorted(set(ts))


def fetch(rel, out, tries=4):
    dst = out / rel
    if dst.exists() and dst.stat().st_size > 0:
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    for k in range(tries):
        try:
            with urllib.request.urlopen(f"{BASE}/{rel}", timeout=120) as r:
                data = r.read()
            tmp = dst.with_suffix(dst.suffix + ".part"); tmp.write_bytes(data); tmp.rename(dst)
            return len(data)
        except Exception as e:
            if k == tries - 1:
                raise RuntimeError(f"{rel}: {e}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", default="conv,sh,gridlock")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--storms", default="", help="comma list; restrict storm registries to these")
    a = ap.parse_args()
    global ONLY
    ONLY = set(a.storms.split(",")) if a.storms else None
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = json.loads(urllib.request.urlopen(f"{BASE}/.zmetadata", timeout=60).read())
    (out / ".zmetadata").write_text(json.dumps(meta))
    rels = []
    for k in meta["metadata"]:
        rels.append(k)                                     # .zgroup/.zattrs/.zarray files
    for v in WHOLE:
        za = meta["metadata"][f"{v}/.zarray"]
        nch = [int(np.ceil(s / c)) for s, c in zip(za["shape"], za["chunks"])]
        for idx in np.ndindex(*nch):
            rels.append(f"{v}/" + ".".join(map(str, idx)))
    ts = wanted(a.jobs.split(","))
    for t in ts:
        i = int((t - T0) / STEP)
        for v in TIME_VARS:
            nd = len(meta["metadata"][f"{v}/.zarray"]["shape"])
            rels.append(f"{v}/{i}" + ".0" * (nd - 1))
    print(f"{len(ts)} timestamps, {len(rels)} objects -> {out}", flush=True)
    if a.dry:
        return
    tot = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for n, b in enumerate(ex.map(lambda r: fetch(r, out), rels)):
            tot += b
            if n % 500 == 0:
                print(f"  {n}/{len(rels)}  {tot/1e9:.1f} GB new", flush=True)
    (out / "STAGED_TIMES.txt").write_text("\n".join(str(t) for t in ts) + "\n")
    print(f"done: {tot/1e9:.1f} GB new", flush=True)


if __name__ == "__main__":
    main()
