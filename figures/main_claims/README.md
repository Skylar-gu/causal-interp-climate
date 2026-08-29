# Main-text figures 1–3

The paper's Figures 1–3 are the `*_notitle.pdf` files here. Each is rendered from the HTML
page beside it (`figure<N>_web_notitle_print.html`), which carries its data inlined in a
`const D = {...}` block, printed with headless chromium and cropped:

```bash
CHROME=~/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome   # any recent chromium
$CHROME --headless=new --no-sandbox --disable-gpu --no-pdf-header-footer \
        --virtual-time-budget=8000 --print-to-pdf=raw.pdf file://$PWD/figure2p5_web_notitle_print.html
pdfcrop --margins 0 raw.pdf figure2p5_interventions_notitle.pdf
```

`build_figure2p5.py` does exactly this for Figure 2 (`CHROME_BIN` overrides the binary): it
derives `figure2p5_web_notitle_print.html` from `figure2_web_notitle_print.html` by replacing the
data block with the spin-feature batteries (`results/skill/mech_3316`, `gain_3316`,
`mech_spin3316`) and swapping the order of the first two bars in panel (c).

`make_figures.py` is the matplotlib fallback: the same three figures (with titles) from the same
shipped files — `savar/results/ladder_gnn/` for Figure 1a, `results/skill/*/verdict.json` and
`results/skill/gain_conv/` for Figure 2, `results/fs_footprints*.npy` for Figure 3.

| file | paper | data |
|---|---|---|
| `figure1_causal_discovery_notitle.pdf` | Fig. 1 | SAVAR ladder (`savar/results/ladder_gnn/`), the eastward-edge audit values and the impulse speeds (bundled in the HTML) |
| `figure2p5_interventions_notitle.pdf` | Fig. 2 | `mech_3316` (a), `gain_3316` (b), `mech_spin3316` / `convection` / `moisture2` medians (c) |
| `figure2_interventions_notitle.pdf` | — (the convection-triplet version of Fig. 2's layout: `convection` (a), `gain_conv` (b)) | |
| `figure3_gridlocked_notitle.pdf` | Fig. 3 | `results/fs_footprints*.npy`, grid-lock ablation scores |
