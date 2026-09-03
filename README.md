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


## Repository structure

graphcast_sae/    GraphCast SAE experiments
  common/         shared utilities and storm metadata
  weights/        pretrained layer-8 SAE
  extraction/     activation extraction
  atlas/          feature labelling and calibration
  storms/         tropical-cyclone interventions
  gridlock/       grid-locked feature analyses and controls
  appendix/       supporting analyses

figures/          paper figures and builders
results/          curated experiment outputs
data/             mesh geometry and feature metadata
docs/prereg/      preregistrations
tests/            self-tests
notebooks/        demo notebook

savar/, obsgraph/, concepts/, heatdome/, legacy/
                  additional and superseded experiments not used in the current paper

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
 
