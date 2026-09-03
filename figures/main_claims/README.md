# Main-text figures

The paper's main-text figures are the `*_notitle.pdf` files here. The **filenames keep the
number they had in an earlier draft** (`figure3_…` is now paper Fig. 1, `figure2_…` is paper
Fig. 2); the `paper` column below is authoritative. Each is rendered from the HTML page beside
it (`figure<N>_web_notitle_print.html`), which carries its data inlined in a `const D = {...}`
block, printed with headless chromium and cropped:

```bash
CHROME=~/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome   # any recent chromium
$CHROME --headless=new --no-sandbox --disable-gpu --no-pdf-header-footer \
        --virtual-time-budget=8000 --print-to-pdf=raw.pdf file://$PWD/figure2_web_notitle_print.html
pdfcrop --margins 0 raw.pdf figure2_interventions_notitle.pdf
```

`build_figure2p5.py` does this for the single-spin-feature variant of the interventions figure
(`CHROME_BIN` overrides the binary): it derives `figure2p5_web_notitle_print.html` from
`figure2_web_notitle_print.html` by replacing the data block with the spin-feature batteries
(`results/skill/mech_3316`, `gain_3316`, `mech_spin3316`) and swapping the order of the first
two bars in panel (c). The paper embeds the convection-triplet `figure2_interventions_notitle.pdf`.

`make_figures.py` is the matplotlib fallback: the same figures (with titles) from the same
shipped files — `results/skill/*/verdict.json` and `results/skill/gain_conv/` for Fig. 2,
`results/fs_footprints*.npy` for Fig. 1.

| file | paper | data |
|---|---|---|
| `figure3_gridlocked_notitle.pdf` | Fig. 1 | `results/fs_footprints*.npy`, grid-lock ablation scores |
| `figure2_interventions_notitle.pdf` | Fig. 2 | `convection` (a), `gain_conv` (b) |
| `figure2p5_interventions_notitle.pdf` | — (single-spin-feature variant of Fig. 2's layout) | `mech_3316` (a), `gain_3316` (b), `mech_spin3316` / `convection` / `moisture2` medians (c) |
| `figure1_causal_discovery_notitle.pdf` | — (SAVAR calibration; not in the current draft) | SAVAR ladder (`savar/results/ladder_gnn/`), the eastward-edge audit values and the impulse speeds (bundled in the HTML) |

The Ida dial-up progression (paper Fig. 3) is built by `figures/paper_fig_ida_dialup.py`, not
from this directory.
