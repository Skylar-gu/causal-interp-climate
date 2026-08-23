"""Flagship pool build — the six decompositions we compare, all at one mode count N.

Consumes the i.i.d. dump (fs_iid_dump.npy) and produces, on the 40,962-node 2to6 mesh:
  leiden_flag, vmax_flag, km_flag  — reference decompositions of the activation field
  sae_flag                          — the CANDIDATE: flagship SAE features clustered into N
  shift_flag, qperm_flag            — the two corrupted anchors (negative controls)
plus per-mode channel directions q_c. Saved as pool_flag_candidates.npy / _channel_dirs.npy
in the same structure the mini analysis code reads.

(needs torch for the SAE; system python3 has it)

Paper: Sec. 4 (pool v1; superseded by build_pool_flag_v2, kept because v2 reuses its footprints)
Inputs: i.i.d. dump (GC_SCRATCH/fs_iid_dump.npy, extraction/extract_iid_dump.py)
Outputs: candidates/pool_flag_candidates.npy; candidates/pool_flag_channel_dirs.npy
Run:   # JAX env, CPU
    python -m graphcast_sae.obsgraph.build_pool
"""
import json, os, sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from graphcast_sae.paths import REPO_ROOT as ROOT, MESH_GEOM, SAE_WEIGHTS, SCRATCH
import graphcast_sae.obsgraph.discover_leiden as dl

DUMP = SCRATCH / "fs_iid_dump.npy"
META = json.load(open(SCRATCH / "fs_iid_meta.json"))
NW, N_MESH, DIM = META["n_windows"], META["n_mesh"], META["dim"]
R_PC = 20
SEED = 0
LEI_RES = float(os.environ.get("LEI_RES", 1.0))
OUT_C = ROOT / "candidates/pool_flag_candidates.npy"
OUT_CD = ROOT / "candidates/pool_flag_channel_dirs.npy"
np.random.seed(SEED)

# ---- mesh geometry (2to6, 40962 nodes) — precomputed in the JAX env ----
def mesh_geometry():
    d = np.load(MESH_GEOM, allow_pickle=True).item()
    return d["lat"], d["lon"], d["xyz"]

# ---- whitened channel-PC field (NW, L, R_PC) ----
def build_field():
    X = np.load(DUMP, mmap_mode="r")                       # (NW*L, DIM) fp16
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(X.shape[0], min(200000, X.shape[0]), replace=False))
    S = np.asarray(X[idx], np.float32)
    mu = S.mean(0)
    pca = PCA(R_PC, random_state=SEED).fit(S - mu)
    comp = pca.components_; whiten = 1.0 / np.sqrt(pca.explained_variance_)
    print(f"[field] channel PCs cumvar={pca.explained_variance_ratio_.sum()*100:.0f}% r={R_PC}", flush=True)
    field = np.empty((NW, N_MESH, R_PC), np.float32)
    for w in range(NW):
        blk = np.asarray(X[w*N_MESH:(w+1)*N_MESH], np.float32) - mu
        field[w] = (blk @ comp.T) * whiten
    return field, dict(mean=mu, components=comp, whiten=whiten)

# ---- reference-decomposition helpers (ported from build_act_candidates) ----
def varimax(Phi, gamma=1.0, q=100, tol=1e-8):
    p, k = Phi.shape; R = np.eye(k); d = 0.0
    for _ in range(q):
        Lm = Phi @ R
        u, s, vt = np.linalg.svd(Phi.T @ (Lm**3 - (gamma/p) * Lm @ np.diag((Lm**2).sum(0))))
        R = u @ vt; dn = s.sum()
        if dn < d * (1 + tol): break
        d = dn
    return Phi @ R

def loading_to_footprint(load):
    if load[np.argmax(np.abs(load))] < 0: load = -load
    fp = np.clip(load, 0, None); m = fp.max()
    if m > 0: fp[fp < 0.05 * m] = 0.0
    s = fp.sum(); return fp / s if s > 0 else fp

def cand_varimax(field, N, L):
    F = field.transpose(1, 0, 2).reshape(L, -1)            # (L, NW*R)
    comp = PCA(N, random_state=SEED).fit(F).components_    # (N, NW*R)
    load = F @ comp.T                                      # (L, N)
    rot = varimax(load)
    return np.stack([loading_to_footprint(rot[:, c]) for c in range(N)])

def cand_kmeans(field, N, L):
    F = field.transpose(1, 0, 2).reshape(L, -1)
    km = KMeans(N, n_init=6, random_state=SEED).fit(F)
    W = np.zeros((N, L), np.float32)
    d = np.linalg.norm(F - km.cluster_centers_[km.labels_], axis=1)
    for c in range(N):
        m = km.labels_ == c; w = np.zeros(L)
        if m.any(): w[m] = 1.0 / (1e-6 + d[m]); w = w / w.sum()
        W[c] = w
    return W

def geometric_knn_shift(base, lat, lon, xyz, dlon=40.0):
    """shift anchor: move each footprint's mass +dlon in longitude (real dynamics, wrong geo)."""
    lon2 = (lon + dlon + 180) % 360 - 180
    latr, lonr = np.radians(lat), np.radians(lon2)
    xyz2 = np.stack([np.cos(latr)*np.cos(lonr), np.cos(latr)*np.sin(lonr), np.sin(latr)], 1)
    from scipy.spatial import cKDTree
    nn = cKDTree(xyz).query(xyz2, k=1)[1]                  # map shifted->nearest original node
    return base[:, nn]

# ---- SAE-cluster member (numpy encode) ----
def build_sae_member(N, k=32):
    z = np.load(SAE_WEIGHTS)                               # the shipped npz
    Wenc = np.asarray(z["W_enc"], np.float32)              # (F,512)  authors' semantics
    bpre = np.asarray(z["b_pre"], np.float32)              # (512,)
    Ffeat = Wenc.shape[0]
    X = np.load(DUMP, mmap_mode="r")
    featmap = np.zeros((Ffeat, N_MESH), np.float32); fire = np.zeros(Ffeat)
    for w in range(NW):
        A = np.asarray(X[w*N_MESH:(w+1)*N_MESH], np.float32)          # (L,512) RAW
        xn = A - A.mean(1, keepdims=True)
        xn = xn / (np.linalg.norm(xn, axis=1, keepdims=True) + 1e-6)  # per-token unit norm
        pre = np.maximum((xn - bpre) @ Wenc.T, 0.0)                   # relu, (L,F)
        idx = np.argpartition(-pre, k, axis=1)[:, :k]                 # TopK per token
        f = np.zeros_like(pre); rows = np.arange(len(A))[:, None]
        f[rows, idx] = pre[rows, idx]
        featmap += f.T; fire += (f > 0).sum(0)
        if w % 40 == 0: print(f"  [sae] encoded {w}/{NW}", flush=True)
    featmap /= NW; alive = fire > 0
    print(f"[sae] alive features {int(alive.sum())}/{Ffeat}", flush=True)
    Fn = featmap[alive]; Fn = Fn / (np.linalg.norm(Fn, axis=1, keepdims=True) + 1e-12)
    emb = PCA(min(50, Fn.shape[0]-1), random_state=SEED).fit_transform(Fn)
    km = KMeans(N, n_init=6, random_state=SEED).fit(emb)
    W = np.stack([loading_to_footprint(featmap[alive][km.labels_ == c].sum(0)) for c in range(N)])
    return W, int(alive.sum())

def main():
    OUT_C.parent.mkdir(exist_ok=True)
    lat, lon, xyz = mesh_geometry()
    field, basis = build_field()
    L = N_MESH

    print(f"[leiden] discover (res={LEI_RES}) on {L} nodes ...", flush=True)
    lei = dl.discover_leiden(field, knn=int(os.environ.get("LEI_KNN", 15)), res=LEI_RES,
                             var_floor_pct=30.0, min_size=int(os.environ.get("LEI_MIN", 40)),
                             max_pix=L, seed=SEED, verbose=True)
    leiden = lei["what"]; N = leiden.shape[0]
    print(f"[leiden] N={N}", flush=True)

    cands = {"leiden_flag": leiden}
    print("[vmax]", flush=True); cands["vmax_flag"] = cand_varimax(field, N, L)
    print("[km]", flush=True);   cands["km_flag"] = cand_kmeans(field, N, L)
    print("[sae]", flush=True);  cands["sae_flag"], n_alive = build_sae_member(N)
    # anchors from vmax (the reliable reference)
    cands["shift_flag"] = geometric_knn_shift(cands["vmax_flag"], lat, lon, xyz, 40.0)
    cands["qperm_flag"] = cands["vmax_flag"].copy()         # geo kept; dynamics broken via q-perm below

    # ---- per-mode channel directions q_c from the dump ----
    X = np.load(DUMP, mmap_mode="r"); cd = {}
    for name, W in cands.items():
        P = np.empty((W.shape[0], NW, DIM), np.float32)
        for w in range(NW):
            A = np.asarray(X[w*N_MESH:(w+1)*N_MESH], np.float32)
            P[:, w, :] = W.astype(np.float32) @ A
        q = np.empty((W.shape[0], DIM), np.float32); mbar = P.mean(1); vf = np.empty(W.shape[0])
        for c in range(W.shape[0]):
            _, s, vt = np.linalg.svd(P[c] - mbar[c], full_matrices=False)
            q[c] = vt[0]; vf[c] = s[0]**2 / (s**2).sum()
        cd[name] = dict(q=q, mbar=mbar, varfrac=vf)
        print(f"  [q] {name} varfrac~{vf.mean():.2f}", flush=True)
    # qperm: permute each mode's channel direction (breaks dynamics, keeps geography)
    rng = np.random.default_rng(SEED)
    cd["qperm_flag"]["q"] = np.stack([cd["vmax_flag"]["q"][c][rng.permutation(DIM)]
                                      for c in range(N)]).astype(np.float32)

    np.save(OUT_C, dict(cands=cands, lat=lat, lon=lon, xyz=xyz,
            provenance=dict(pool="flagship", N=int(N), mesh="2to6", n_iid=NW,
                            sae_alive=n_alive, members=list(cands))), allow_pickle=True)
    np.save(OUT_CD, cd, allow_pickle=True)
    print(f"\nDONE N={N}  members={list(cands)}  -> {OUT_C.name}, {OUT_CD.name}", flush=True)

if __name__ == "__main__":
    main()
