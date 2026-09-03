# Causal discovery and interventions inside weather forecasting models 

## Quick start — demo

```bash
pip install -r requirements.txt     # demo + figure deps; GPU block is commented
jupyter notebook notebooks/demo.ipynb     # or: run all cells
```

`notebooks/demo.ipynb` runs on **CPU in a few minutes** and regenerates, from the
result files shipped here:

1. the Ida dial-up progression figure (each calibrated genesis-ingredient group
   doubled, watched on the model's internal cyclone feature at +12/+24/+36/+48 h),
2. the seven-storm mechanism-ablation table (deepening lost vs ERA5) for the
   convection triplet, the low-level-spin group, and the moisture group,
3. the gain-sweep curves for the convection- and vorticity-associated features (Ida / Haishen / Patricia),
4. the grid-locked features: world maps of their footprints, single-feature and
   group ablation tables against matched controls,


## Layout

The code is two experiment lanes. The current paper is the **interventions + grid-locked
features** lane; the **causal-discovery** lane (SAVAR ground truth, the observational PCMCI+
graph) is kept for provenance but is not in the current draft.

```
graphcast_sae/    experiment code, one subpackage per experiment (run as
                  `python -m graphcast_sae.<group>.<script>` from the repo root)

  # shared
  common/         shared machinery (fs_common), storm registries, signature physics
  weights/        the published SAE (TopK, k=32, dict 4096, layer 8) + config
  extraction/     layer-8 activation dumps (mini_* = the graphcast_small lane)

  # interventions + grid-locked features  — the current paper
  atlas/          feature labelling / calibration (mechanism / physics atlas)
  storms/         seven-storm mechanism ablations, Ida dial-up, gain sweep
  gridlock/       grid-locked features: scores, matched controls, ablations
  appendix/       supporting analyses (parity, mediation, locality, skill)

  # causal discovery  — not in the current draft
  obsgraph/       observational graph (pool, 12-yr trajectory, PCMCI+ hybrid)
  concepts/       purified concept groups and their response operators

  # other
  heatdome/       2021 heat-dome blocking study (results shipped; not in the paper)
  legacy/         superseded lanes kept for provenance (steering, mega battery)

savar/            causal-discovery lane: SAVAR benchmark, CNN/GNN forecasters, SAE ladders
docs/prereg/      the pre-registrations the scripts cite; docs/notes/ the cited notes
tests/            CPU self-tests
figures/          the paper's figure PDFs and their builders (figures/main_claims/ holds
                  the web-rendered print PDFs + the HTML they render from)
results/          curated result files (verdicts, sweeps, scores, ablation arms)
data/             mesh geometry, per-feature footprints, land-sea mask
notebooks/        demo.ipynb (executed)
```

## Running the GPU experiments

The larger GraphCast rollouts experiments need one
~46 GB GPU and three public assets not shipped here:

- **GraphCast weights** (0.25°, 37 levels): from the DeepMind GraphCast release.
- **ERA5 / WeatherBench2**: streamed from the public GCS bucket
  (`gs://weatherbench2/...`); scripts read it via anonymous `gcsfs`.
- **The i.i.d. activation dump** (6.7 GB): regenerate with the dump script in
  `graphcast_sae/extraction/extract_iid_dump.py` (160 windows, layer-8 mesh activations); every script that
  needs it names its path.

The SAE weights are shipped (`graphcast_sae/weights/sae_k32_lat4096_lay08.npz`, 17 MB)
and public (`theodoremacmillan/sae-graphcast-k32-lat4096-lay08`).
 
