# Causal discovery and interventions inside GraphCast 

## Quick start — demo

```bash
pip install -r requirements.txt     # demo + figure deps; GPU block is commented
jupyter notebook notebooks/demo.ipynb     # or: run all cells
```

`notebooks/demo.ipynb` runs on **CPU in a few minutes** and regenerates, from the
result files shipped here:

1. the Ida dial-up progression figure (each genesis ingredient doubled, watched
   on the model's internal cyclone feature at +12/+24/+36/+48 h),
2. the seven-storm mechanism-ablation table (deepening lost vs ERA5),
3. the convection gain-sweep curves (Ida / Haishen / Patricia, with optima),
4. the grid-locked features: world maps of their footprints, single-feature and
   group ablation tables against matched controls,


## Layout

```
graphcast_sae/    the experiment code, one subpackage per experiment (run as
                  `python -m graphcast_sae.<group>.<script>` from the repo root)
  common/         shared machinery (fs_common), storm registries, signature physics
  weights/        the published SAE (TopK, k=32, dict 4096, layer 8) + config
  extraction/     layer-8 activation dumps (mini_* = the graphcast_small lane)
  storms/         seven-storm mechanism ablations, Ida dial-up, gain sweep
  gridlock/       grid-locked features: scores, matched controls, ablations
  atlas/          feature labelling (mechanism / physics atlas)
  concepts/       purified concept groups and their response operators
  obsgraph/       observational graph (pool, 12-yr trajectory, PCMCI+ hybrid)
  appendix/       negative-appendix analyses (parity, mediation, locality, skill)
  heatdome/       2021 heat-dome blocking study (results shipped; not in the paper)
  legacy/         superseded lanes kept for provenance (steering, mega battery)
docs/prereg/      the pre-registrations the scripts cite; docs/notes/ the cited notes
tests/            CPU self-tests
figures/          figure builders whose inputs ship in results/, plus the paper's PDFs
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
 
