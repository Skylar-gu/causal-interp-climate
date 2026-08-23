# `graphcast_sae.extraction` — layer-8 activation dumps

Infrastructure for every experiment: one teacher-forced GraphCast forward yields one
512-d embedding per M6 mesh node (40,962 nodes) at processor step 9 (the authors'
"layer0008"), which the published SAE encodes. Outputs land under `GC_SCRATCH`
(`graphcast_sae/paths.py`), never in git.

Run order (JAX env; GPU for the flagship model):

1. `extract_iid_dump.py` — 160 i.i.d. 3-frame windows (2016–2020) → `fs_iid_dump.npy`
   (6.7 GB fp16) + `fs_iid_meta.json`. ~4 s/window on a 46 GB GPU. Every atlas,
   grid-lock, concept and control script reads this dump.
2. `fs_extract.py` → `fs_catalog.py` — the older 24-window (2021) extraction and its
   encoded catalog (`fs_catalog.npz`); only the `legacy/` retry suite reads them.

`mini_*` modules are the **graphcast_small** (1°, 13-level, 10,242-node) lane —
`mini_extract_layer8.py`, `mini_extract_traj.py`, `mini_extract_wb2.py`,
`mini_wb2_stream.py`. They are the small-model prototypes of the flagship extractors
and are not used by any paper result; kept so the lane is reproducible.

External assets: GraphCast params (`GRAPHCAST_PARAMS`), DeepMind `stats/`
(`GRAPHCAST_ASSETS`), WeatherBench2 ERA5 streamed from the public GCS bucket.
