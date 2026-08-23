# `graphcast_sae.atlas` — what each SAE feature tracks

Paper: Sec. 3 — the mechanism labels (convection, moisture, vorticity, shear, …) that define
the intervention groups in Table `tab:mechanism-interventions`; `figures/paper_fig_exposure.py`.
Shipped outputs: `results/fs_mechanisms.npy` (v1), `results/fs_mechanisms_v2.npy` (calibrated).

Run order (JAX env, CPU; every script reads the i.i.d. dump and streams ERA5 fields at the
dump windows from WeatherBench2):

1. `label_expanded.py` — z-scores of every feature against physical / geographic / temporal
   references → `fs_atlas.npy`; `label_extra.py` adds blocking, atmospheric-river, ENSO,
   baroclinicity detectors → `fs_atlas_extra.npy`
2. `label_mechanisms.py` (v1) → `label_mechanisms_v2.py` (rotation-null calibrated) →
   `label_rescore.py` (empirical p-values) → `fs_mechanisms_v2.npy`; `label_banded.py` is the
   latitude-banded repair used for the concept groups
3. `feature_select.py` — per-feature catalog (firing rate, footprint, coherence, centroid)
   → `candidates/fs_feature_catalog.npy` (torch env; regenerable, not shipped)
4. `atlas_analyze.py`, `atlas_classify.py` — census and taxonomy of the dictionary
5. `label_verify_score.py` — verify labels by intervention from runs already on disk

`fs_atlas*.npy` and the catalog are not shipped (regenerable from the dump).
