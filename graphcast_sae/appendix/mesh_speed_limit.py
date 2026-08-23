"""GraphCast's architectural information-propagation speed limit.

Pure CPU / pure geometry. No forward pass, no GPU, no weights beyond the two
scalars read out of the checkpoint header (`mesh_size`, `gnn_msg_steps`).

What it computes
----------------
1. The multi-mesh: `icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere`
   with `splits = model_config.mesh_size`, merged exactly as
   `GraphCast._init_mesh_graph` does (`merge_meshes` -> `faces_to_edges`).
2. Great-circle length of every mesh<->mesh edge, per refinement level.
3. k-hop reachability on the mesh graph with k = `model_config.gnn_msg_steps`
   (the processor applies that many *unshared* message-passing steps per model
   step; each step is one hop -- see `deep_typed_graph_net._process`).
4. The grid->mesh and mesh->grid reach, reported SEPARATELY, not folded in.
5. The comparison against the physics bands and the observed edge speeds.

(only numpy/scipy are imported from the graphcast package; jax is never touched)

Paper: Appendix app:mesh (Table tab:hops; the 79-106 m/s band)
Inputs: results/skill/convection/run_ida2021.npy (shipped); GraphCast params (GRAPHCAST_PARAMS)
Outputs: results/mesh_ecc_all.npy; results/mesh_speed_limit.json
Run:   # JAX env, CPU
    python -m graphcast_sae.appendix.mesh_speed_limit
"""
import json
import pathlib
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg

from graphcast_sae.paths import REPO_ROOT as ROOT, GRAPHCAST_PARAMS

from graphcast import icosahedral_mesh as im  # noqa: E402  (numpy+scipy only)

CKPT = GRAPHCAST_PARAMS
OUT = ROOT / "results/mesh_speed_limit.json"

R_EARTH_KM = 6371.0          # same radius local_physics.json used (verified)
STEP_SECONDS = 6 * 3600.0    # one autoregressive model step
N_SOURCES = 200

def gc_km(u, v):
    """Great-circle distance (km) between unit vectors; u (n,3) or (3,), v (m,3)."""
    d = np.clip(np.atleast_2d(u) @ np.atleast_2d(v).T, -1.0, 1.0)
    return np.arccos(d) * R_EARTH_KM

def cart_to_latlon(v):
    lon = np.mod(np.rad2deg(np.arctan2(v[:, 1], v[:, 0])), 360.0)
    lat = 90.0 - np.rad2deg(np.arccos(np.clip(v[:, 2], -1, 1)))
    return lat, lon

def q(a):
    a = np.asarray(a, float)
    return dict(min=float(a.min()), p25=float(np.percentile(a, 25)),
                median=float(np.median(a)), p75=float(np.percentile(a, 75)),
                max=float(a.max()), mean=float(a.mean()))

def main():
    rep = {}

    # ---------------------------------------------------- 0. config ---------
    z = np.load(CKPT, allow_pickle=True)
    cfg = {k.split(":", 1)[1]: z[k].item() for k in z.files
           if k.startswith("model_config:")}
    mesh_size = int(cfg["mesh_size"])
    K = int(cfg["gnn_msg_steps"])
    rqf = float(cfg["radius_query_fraction_edge_length"])
    print(f"[config] {CKPT.name}")
    print(f"  mesh_size (splits)                  = {mesh_size}")
    print(f"  gnn_msg_steps (processor hops/step) = {K}")
    print(f"  radius_query_fraction_edge_length   = {rqf:.10f}")
    print(f"  resolution                          = {cfg['resolution']}")
    rep["config"] = dict(mesh_size=mesh_size, gnn_msg_steps=K,
                         radius_query_fraction_edge_length=rqf,
                         resolution=float(cfg["resolution"]),
                         step_seconds=STEP_SECONDS,
                         source="checkpoint header graphcast_flagship_0p25_37lev.npz")

    # ---------------------------------------------------- 1. the mesh -------
    meshes = im.get_hierarchy_of_triangular_meshes_for_sphere(splits=mesh_size)
    merged = im.merge_meshes(meshes)
    V = np.asarray(merged.vertices, dtype=np.float64)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    n_nodes = V.shape[0]
    lat, lon = cart_to_latlon(V)

    print(f"\n[mesh] refinement levels present = {len(meshes)} "
          f"(level 0 = icosahedron .. level {mesh_size})")
    print(f"[mesh] finest-mesh nodes = {n_nodes}")

    # per-level edges (each level's faces contribute its own edge set)
    lvl_rows = []
    all_s, all_r, all_lvl = [], [], []
    undirected_seen = {}
    for l, m in enumerate(meshes):
        s, r = im.faces_to_edges(np.asarray(m.faces))
        L = gc_km(V[s], V[r]).diagonal() if False else \
            np.arccos(np.clip(np.sum(V[s] * V[r], axis=1), -1, 1)) * R_EARTH_KM
        key = np.stack([np.minimum(s, r), np.maximum(s, r)], 1)
        uk = {tuple(k) for k in key}
        newk = {k for k in uk if k not in undirected_seen}
        for k in newk:
            undirected_seen[k] = l
        lvl_rows.append(dict(
            level=l, n_vertices=int(m.vertices.shape[0]),
            n_faces=int(m.faces.shape[0]),
            n_directed_edges=int(s.shape[0]),
            n_undirected_edges=len(uk),
            n_undirected_new=len(newk),
            len_km=q(L),
        ))
        all_s.append(s); all_r.append(r)
        all_lvl.append(np.full(s.shape[0], l, dtype=np.int8))

    S = np.concatenate(all_s); Rc = np.concatenate(all_r)
    ELVL = np.concatenate(all_lvl)
    ELEN = np.arccos(np.clip(np.sum(V[S] * V[Rc], axis=1), -1, 1)) * R_EARTH_KM

    print(f"[mesh] merged multi-mesh directed edges = {S.shape[0]}  "
          f"(unique undirected = {len(undirected_seen)})")
    print("\n  level  n_vert   n_faces  dir_edges  undir_edges  new_undir   "
          "min_km   med_km   max_km")
    for row in lvl_rows:
        d = row["len_km"]
        print(f"  {row['level']:>5}  {row['n_vertices']:>6}  {row['n_faces']:>8}  "
              f"{row['n_directed_edges']:>9}  {row['n_undirected_edges']:>11}  "
              f"{row['n_undirected_new']:>9}  "
              f"{d['min']:>7.1f}  {d['median']:>7.1f}  {d['max']:>7.1f}")
    print(f"  OVERALL max mesh edge = {ELEN.max():.1f} km   "
          f"median = {np.median(ELEN):.1f} km   min = {ELEN.min():.1f} km")
    rep["levels"] = lvl_rows
    rep["mesh"] = dict(n_nodes=n_nodes, n_directed_edges=int(S.shape[0]),
                       n_undirected_edges=len(undirected_seen),
                       edge_len_km_all=q(ELEN))

    # ---------------------------------------------------- 1b. cross-check ---
    snap = ROOT / "results/skill/convection/run_ida2021.npy"
    if snap.exists():
        d = np.load(snap, allow_pickle=True).item()["snap"]["baseline_mid"]
        mlat = np.asarray(d["mlat"], float); mlon = np.asarray(d["mlon"], float)
        ok_n = mlat.shape[0] == n_nodes
        if ok_n:
            dlat = np.abs(mlat - lat).max()
            dlon = np.abs((mlon - lon + 180) % 360 - 180).max()
            # positional agreement in km
            vs = np.stack([
                np.cos(np.deg2rad(mlat)) * np.cos(np.deg2rad(mlon)),
                np.cos(np.deg2rad(mlat)) * np.sin(np.deg2rad(mlon)),
                np.sin(np.deg2rad(mlat))], 1)
            dkm = np.arccos(np.clip(np.sum(vs * V, 1), -1, 1)) * R_EARTH_KM
            print(f"\n[cross-check vs stored mlat/mlon] n={mlat.shape[0]} "
                  f"max|dlat|={dlat:.3e} deg  max|dlon|={dlon:.3e} deg  "
                  f"max node offset = {dkm.max():.4f} km")
            rep["crosscheck"] = dict(n=int(mlat.shape[0]), matched=True,
                                     max_dlat_deg=float(dlat),
                                     max_dlon_deg=float(dlon),
                                     max_offset_km=float(dkm.max()))
        else:
            print(f"\n[cross-check] MISMATCH: stored n={mlat.shape[0]} vs {n_nodes}")
            rep["crosscheck"] = dict(n=int(mlat.shape[0]), matched=False)

    # ---------------------------------------------------- 2. k-hop BFS ------
    A = sp.csr_matrix((np.ones(S.shape[0], np.int8), (S, Rc)),
                      shape=(n_nodes, n_nodes))

    # node creation level (nested vertex ordering)
    nv = np.array([m.vertices.shape[0] for m in meshes])
    node_level = np.searchsorted(nv, np.arange(n_nodes), side="right")

    # 200 sources spread over the globe: nearest mesh node to a Fibonacci sphere
    i = np.arange(N_SOURCES) + 0.5
    phi = np.arccos(1 - 2 * i / N_SOURCES)
    gold = np.pi * (1 + 5 ** 0.5) * i
    fib = np.stack([np.cos(gold) * np.sin(phi), np.sin(gold) * np.sin(phi),
                    np.cos(phi)], 1)
    srcs = np.unique(np.argmax(fib @ V.T, axis=1))
    print(f"\n[bfs] {len(srcs)} globally-spread source nodes; "
          f"creation-level histogram = "
          f"{np.bincount(node_level[srcs], minlength=mesh_size+1).tolist()}")
    print(f"[bfs] whole-mesh creation-level histogram = "
          f"{np.bincount(node_level, minlength=mesh_size+1).tolist()}")

    hops = csg.dijkstra(A, directed=True, unweighted=True, indices=srcs)
    dist = np.arccos(np.clip(V[srcs] @ V.T, -1, 1)) * R_EARTH_KM

    radius_k = np.array([dist[j][hops[j] <= K].max() for j in range(len(srcs))])
    frac_k = np.array([(hops[j] <= K).mean() for j in range(len(srcs))])
    ecc = hops.max(axis=1)

    print(f"[bfs] hops needed to reach EVERY mesh node (graph eccentricity of "
          f"the sampled sources): min={int(ecc.min())} median={np.median(ecc):.1f} "
          f"max={int(ecc.max())}")

    print(f"\n[k-hop radius, k = {K}]  max great-circle distance reached in one "
          f"6-h model step")
    rk = q(radius_k)
    for name in ("min", "p25", "median", "p75", "max", "mean"):
        print(f"    {name:>6}: {rk[name]:>9.1f} km   "
              f"= {rk[name]*1000/STEP_SECONDS:>7.1f} m/s")
    print(f"    fraction of the 40962 mesh nodes reachable within {K} hops: "
          f"min={frac_k.min():.4f} median={np.median(frac_k):.4f} "
          f"max={frac_k.max():.4f}")

    # hop-by-hop growth curve (median over sources)
    curve = []
    for k in range(0, K + 1):
        rr = np.array([dist[j][hops[j] <= k].max() for j in range(len(srcs))])
        ff = np.array([(hops[j] <= k).mean() for j in range(len(srcs))])
        curve.append(dict(k=k, median_radius_km=float(np.median(rr)),
                          max_radius_km=float(rr.max()),
                          median_frac_reached=float(np.median(ff))))
    print(f"\n[growth]  k : median radius (km) : max radius (km) : median frac nodes")
    for c in curve:
        print(f"   {c['k']:>3} : {c['median_radius_km']:>12.1f} : "
              f"{c['max_radius_km']:>12.1f} : {c['median_frac_reached']:>10.4f}")

    # stratified by creation level of the source
    strat = {}
    for l in range(mesh_size + 1):
        cand = np.where(node_level == l)[0]
        rng = np.random.default_rng(0)
        pick = cand if len(cand) <= 40 else rng.choice(cand, 40, replace=False)
        h = csg.dijkstra(A, directed=True, unweighted=True, indices=pick)
        dd = np.arccos(np.clip(V[pick] @ V.T, -1, 1)) * R_EARTH_KM
        rr = np.array([dd[j][h[j] <= K].max() for j in range(len(pick))])
        strat[l] = dict(n=int(len(pick)), radius_km=q(rr),
                        speed_ms=q(rr * 1000 / STEP_SECONDS))
    print(f"\n[stratified by source creation level]  radius in {K} hops")
    print("   level    n   min_km   med_km   max_km   med_m/s   max_m/s")
    for l, s_ in strat.items():
        r_ = s_["radius_km"]
        print(f"   {l:>5}  {s_['n']:>3}  {r_['min']:>7.1f}  {r_['median']:>7.1f}  "
              f"{r_['max']:>7.1f}  {r_['median']*1000/STEP_SECONDS:>8.1f}  "
              f"{r_['max']*1000/STEP_SECONDS:>8.1f}")

    rep["khop"] = dict(k=K, n_sources=int(len(srcs)),
                       radius_km=rk,
                       speed_ms=q(radius_k * 1000 / STEP_SECONDS),
                       frac_nodes_reached=q(frac_k),
                       eccentricity_hops=q(ecc),
                       growth_curve=curve,
                       by_source_level={str(k_): v for k_, v in strat.items()})

    # ------------------------------------------- 2b. exact graph diameter ---
    # 16 hops covers the globe, so the informative number is the INVERSE:
    # how many hops the mesh actually needs. Exact eccentricity over all nodes.
    ecc_cache = ROOT / "results/mesh_ecc_all.npy"
    if ecc_cache.exists():
        ecc_all = np.load(ecc_cache)
        assert ecc_all.shape[0] == n_nodes
    else:
        ecc_all = np.zeros(n_nodes, dtype=np.int16)
        CH = 1024
        for a in range(0, n_nodes, CH):
            b = min(a + CH, n_nodes)
            h = csg.dijkstra(A, directed=True, unweighted=True,
                             indices=np.arange(a, b))
            ecc_all[a:b] = h.max(axis=1).astype(np.int16)
        ecc_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(ecc_cache, ecc_all)
    diam = int(ecc_all.max())
    print(f"\n[diameter] EXACT eccentricity over all {n_nodes} mesh nodes: "
          f"min={int(ecc_all.min())} median={np.median(ecc_all):.1f} "
          f"max (= graph diameter) = {diam}")
    print(f"[diameter] processor depth K={K} vs diameter {diam}  ->  "
          f"spare depth = {K - diam} hops "
          f"({K/diam:.2f}x more message passing than needed to cross the globe)")
    print(f"[diameter] eccentricity histogram: "
          f"{np.bincount(ecc_all.astype(int)).tolist()}")
    rep["diameter"] = dict(exact_diameter_hops=diam,
                           eccentricity=q(ecc_all),
                           histogram=np.bincount(ecc_all.astype(int)).tolist(),
                           spare_depth_hops=K - diam,
                           depth_over_diameter=K / diam)

    # ------------------------------------------- 2c. hops needed per band ---
    # For each physics band speed v, the distance it covers in 6 h; how many
    # hops does the mesh need to cover that, and what fraction of the 16-step
    # budget is that?
    bands = [("TC translation 5 m/s", 5.0), ("TC translation 10 m/s", 10.0),
             ("first baroclinic 50 m/s", 50.0),
             ("repo IMPOSSIBLE floor 150 m/s", 150.0),
             ("fastest observed edge 359.5 m/s", 359.5),
             ("antipodal 926.6 m/s", 926.6)]
    print(f"\n[hops needed]  hops the mesh needs to move information as fast as "
          f"each band (median over {len(srcs)} sources)")
    print("   band                                dist in 6h (km)   hops needed"
          "   % of the 16-hop budget")
    hop_rows = []
    for name, v in bands:
        dkm_ = v * STEP_SECONDS / 1000.0
        need = []
        for j in range(len(srcs)):
            reach = np.array([dist[j][hops[j] <= k].max() for k in range(K + 1)])
            idx = np.argmax(reach >= min(dkm_, 20015.0))
            need.append(idx if reach[idx] >= min(dkm_, 20015.0) else K + 1)
        need = np.array(need)
        med = float(np.median(need))
        print(f"   {name:<34}{dkm_:>14.1f}   {med:>11.1f}   {100*med/K:>20.1f}%")
        hop_rows.append(dict(band=name, speed_ms=v, dist_km=dkm_,
                             hops_needed=q(need), median_hops=med,
                             pct_of_budget=100 * med / K))
    rep["hops_needed"] = hop_rows

    # ------------------------------------------- 2d. one-hop reach ----------
    onehop = np.zeros(n_nodes)
    np.maximum.at(onehop, S, ELEN)
    print(f"\n[one hop] longest edge incident to a node (=1-step reach), km: "
          f"min={onehop.min():.1f} median={np.median(onehop):.1f} "
          f"p95={np.percentile(onehop,95):.1f} max={onehop.max():.1f}")
    print(f"[one hop] implied m/s over a 6-h step: "
          f"median={np.median(onehop)*1000/STEP_SECONDS:.1f} "
          f"max={onehop.max()*1000/STEP_SECONDS:.1f}")
    print(f"[one hop] fraction of mesh nodes whose single-hop reach already "
          f"exceeds 150 m/s x 6h (3240 km): "
          f"{(onehop > 3240).mean():.4f}")
    rep["one_hop"] = dict(reach_km=q(onehop),
                          speed_ms=q(onehop * 1000 / STEP_SECONDS),
                          frac_nodes_over_150ms=float((onehop > 3240).mean()))

    # ------------------------------------------- 2e. edges above a band -----
    und = {}
    for (a, b), l in undirected_seen.items():
        und[(a, b)] = l
    ua = np.array([k[0] for k in und]); ub = np.array([k[1] for k in und])
    ul = np.array(list(und.values()))
    ulen = np.arccos(np.clip(np.sum(V[ua] * V[ub], 1), -1, 1)) * R_EARTH_KM
    print(f"\n[long edges] undirected mesh edges whose SINGLE hop already "
          f"exceeds a band (dist = v x 6 h):")
    long_rows = []
    for name, v in [("50 m/s (1080 km)", 50.0), ("100 m/s (2160 km)", 100.0),
                    ("150 m/s (3240 km)", 150.0), ("300 m/s (6480 km)", 300.0)]:
        thr = v * STEP_SECONDS / 1000.0
        m = ulen > thr
        lv = sorted(set(ul[m].tolist()))
        print(f"   > {name:<20} n={int(m.sum()):>6} / {len(und)}  "
              f"({100*m.mean():.4f}%)   from levels {lv}")
        long_rows.append(dict(band=name, speed_ms=v, n_edges=int(m.sum()),
                              frac=float(m.mean()), levels=lv))
    rep["long_edges"] = dict(n_undirected=len(und), rows=long_rows)

    # ------------------------------------------- 2f. finest-mesh-only -------
    keep = ELVL == mesh_size
    Af = sp.csr_matrix((np.ones(int(keep.sum()), np.int8),
                        (S[keep], Rc[keep])), shape=(n_nodes, n_nodes))
    hf = csg.dijkstra(Af, directed=True, unweighted=True, indices=srcs)
    rf = np.array([dist[j][hf[j] <= K].max() for j in range(len(srcs))])
    print(f"\n[counterfactual: FINEST MESH ONLY, no multi-mesh long edges] "
          f"{K}-hop radius")
    qf = q(rf)
    print(f"   min={qf['min']:.1f} km  median={qf['median']:.1f} km  "
          f"max={qf['max']:.1f} km")
    print(f"   -> median {qf['median']*1000/STEP_SECONDS:.1f} m/s, "
          f"max {qf['max']*1000/STEP_SECONDS:.1f} m/s")
    print(f"   multi-mesh speed-up over a single-resolution mesh: "
          f"{rk['median']/qf['median']:.2f}x")
    rep["finest_only"] = dict(radius_km=qf,
                              speed_ms=q(rf * 1000 / STEP_SECONDS),
                              multimesh_speedup=float(rk["median"] / qf["median"]))

    # theoretical single-hop bound: longest edge / step
    print(f"\n[single hop] longest mesh edge = {ELEN.max():.1f} km "
          f"-> {ELEN.max()*1000/STEP_SECONDS:.1f} m/s if used once in a step")
    print(f"[16 hops of the longest edge, ignoring geometry] "
          f"{K*ELEN.max():.0f} km >> half-circumference "
          f"{np.pi*R_EARTH_KM:.0f} km -- so the bound is saturated by the sphere")

    # ---------------------------------------------------- 3. grid<->mesh ----
    finest = meshes[-1]
    fs, fr = im.faces_to_edges(np.asarray(finest.faces))
    chord = np.linalg.norm(np.asarray(finest.vertices)[fs]
                           - np.asarray(finest.vertices)[fr], axis=-1)
    max_chord = float(chord.max())                     # == _get_max_edge_distance
    query_radius_chord = max_chord * rqf
    query_radius_km = 2 * np.arcsin(min(query_radius_chord / 2, 1.0)) * R_EARTH_KM
    max_fine_edge_km = float(2 * np.arcsin(np.clip(chord / 2, 0, 1)).max() * R_EARTH_KM)
    print(f"\n[grid->mesh encoder] radius query = {rqf:.4f} x max finest chord "
          f"({max_chord:.6f} unit) = {query_radius_chord:.6f} unit "
          f"= {query_radius_km:.1f} km great-circle")
    print(f"[mesh->grid decoder] each grid node reads the 3 vertices of its "
          f"containing finest-mesh triangle; bound = longest level-{mesh_size} "
          f"edge = {max_fine_edge_km:.1f} km")
    total_km = query_radius_km + rk["max"] + max_fine_edge_km
    print(f"[end-to-end, grid->grid, one 6-h step] "
          f"{query_radius_km:.1f} + {rk['max']:.1f} + {max_fine_edge_km:.1f} "
          f"= {total_km:.1f} km -> {total_km*1000/STEP_SECONDS:.1f} m/s "
          f"(capped in practice by the sphere: 20015 km / 6h = 926.6 m/s)")
    rep["grid_mesh"] = dict(
        grid2mesh_radius_km=float(query_radius_km),
        grid2mesh_radius_unit_chord=float(query_radius_chord),
        mesh2grid_radius_km=float(max_fine_edge_km),
        end_to_end_km=float(total_km),
        end_to_end_ms=float(total_km * 1000 / STEP_SECONDS),
        antipodal_cap_ms=float(np.pi * R_EARTH_KM * 1000 / STEP_SECONDS))

    # ---------------------------------------------------- 4. comparison -----
    v_med = rk["median"] * 1000 / STEP_SECONDS
    v_max = rk["max"] * 1000 / STEP_SECONDS
    print("\n" + "=" * 78)
    print("COMPARISON  (mesh speed limit = per-6h-step k-hop radius / 21600 s)")
    print("=" * 78)
    rows = [
        ("TC translation (slow)", 5.0),
        ("TC translation (fast)", 10.0),
        ("first baroclinic mode", 50.0),
        ("repo 'admissible' ceiling", 50.0),
        ("repo 'IMPOSSIBLE' floor", 150.0),
        ("graph_v2 pcmci min edge", 14.7),
        ("graph_v2 pcmci median edge", 15.6),
        ("graph_v2 pcmci max edge", 16.6),
        ("local_physics pcmci median edge", 119.4),
        ("local_physics pcmci max edge", 315.3),
        ("local_physics lpcmci median edge", 108.2),
        ("local_physics lpcmci max edge", 359.5),
        ("SAE co-encoding artefact band lo", 79.0),
        ("SAE co-encoding artefact band hi", 106.0),
    ]
    print(f"{'quantity':<34}{'m/s':>9}  {'x median mesh':>14}  inside mesh reach?")
    cmp_rows = []
    for name, v in rows:
        inside = "YES" if v <= v_med else ("yes (tail only)" if v <= v_max else "NO")
        print(f"{name:<34}{v:>9.1f}  {v/v_med:>14.3f}  {inside}")
        cmp_rows.append(dict(quantity=name, speed_ms=v,
                             ratio_to_median_mesh=v / v_med, inside=inside))
    print(f"{'MESH LIMIT (median source)':<34}{v_med:>9.1f}  {1.0:>14.3f}  --")
    print(f"{'MESH LIMIT (max source)':<34}{v_max:>9.1f}  "
          f"{v_max/v_med:>14.3f}  --")
    print("\n[the gap]")
    for name, v in [("TC translation 10 m/s", 10.0),
                    ("first baroclinic 50 m/s", 50.0),
                    ("repo IMPOSSIBLE floor 150 m/s", 150.0),
                    ("fastest observed edge 359.5 m/s", 359.5)]:
        print(f"  mesh / {name:<34} = {v_med/v:>8.2f}x (median), "
              f"{v_max/v:>8.2f}x (max)")
    rep["comparison"] = dict(mesh_median_ms=v_med, mesh_max_ms=v_max,
                             rows=cmp_rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2))
    print(f"\n[written] {OUT}")

if __name__ == "__main__":
    main()
