"""Figure 2.5: Figure 2's layout with the storm-core spin feature 3316 in panels (a) and (b), and
panel (c) with the spin and convection bars swapped.

Derives figure2p5_web_notitle_print.html from figure2_web_notitle_print.html by replacing the
inlined D.fig2 data block and the panel titles/labels, then renders it with headless chromium and
pdfcrop (same pipeline as Figure 2; see README.md).

    python3 figures/main_claims/build_figure2p5.py            # build html + pdf
    python3 figures/main_claims/build_figure2p5.py --no-pdf   # html only
"""
import json, os, re, shutil, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = HERE / "figure2_web_notitle_print.html"
DST = HERE / "figure2p5_web_notitle_print.html"
PDF = HERE / "figure2p5_interventions_notitle.pdf"
# any recent chromium/chrome: CHROME_BIN=<path>, else whatever is on PATH
CHROME = Path(os.environ.get("CHROME_BIN") or shutil.which("chromium") or shutil.which("chromium-browser")
              or shutil.which("google-chrome") or shutil.which("chrome") or "chromium")
STORMS = ["ida2021", "michael2018", "haishen2020", "goni2020", "haiyan2013", "patricia2015", "wilma2005"]
GAINS = [0., 1.25, 1.5, 1.75, 2., 2.5, 3.]
ABL = "mech_3316"          # panel (a): single-feature seven-storm battery
GAIN = "gain_3316"         # panel (b): gain sweep
SPIN_GROUP = "mech_spin3316"   # panel (c) spin bar (the group, as in Figure 2 after 2026-08-29)


def effects(directory):
    d = json.load(open(ROOT / "results/skill" / directory / "verdict.json"))["metrics"]
    return ([d[s]["arms"]["conv-normal"]["d_deepen"] for s in STORMS],
            [d[s]["arms"]["rand-normal"]["d_deepen"] for s in STORMS])


def gain_curve(storm):
    truth = np.load(ROOT / f"results/skill/{GAIN}/era5_truth.npy", allow_pickle=True).item()[storm]["mslp_min"]
    res = np.load(ROOT / f"results/skill/{GAIN}/run_{storm}.npy", allow_pickle=True).item()["res"]
    win = max(int(np.argmin(truth)), 6) + 1
    def rmse(arm):
        m = np.asarray(res[arm]["mslp_min"]); n = min(win, len(m), len(truth))
        return float(np.sqrt(np.mean((m[:n] - np.asarray(truth)[:n]) ** 2)))
    return dict(gains=GAINS, err=[rmse("gain-%g" % g) for g in GAINS], baseline=rmse("baseline"))


def main(render=True):
    html = SRC.read_text()
    m = re.search(r'const D = (\{.*?\});\n', html, re.S)
    D = json.loads(m.group(1))
    f2 = D["fig2"]
    conv, ctrl = effects(ABL)
    f2["conv"], f2["ctrl"] = conv, ctrl
    f2["gain"] = {s: gain_curve(s) for s in ("ida2021", "haishen2020")}
    groups = f2["groups"]
    spin = [g for g in groups if g["label"] == "low-level spin"][0]
    convg = [g for g in groups if g["label"].startswith("convection")][0]
    moist = [g for g in groups if g["label"].startswith("q600")][0]
    f2["groups"] = [spin, convg, moist]
    html = html[:m.start(1)] + json.dumps(D) + html[m.end(1):]

    # panel (b) y-axis: keep Figure 2's 0-21 unless the spin curve exceeds it
    top = max(v for g in f2["gain"].values() for v in g["err"] + [g["baseline"]])
    if top > 20.5:
        hi = int(np.ceil(top / 5.0) * 5)
        ticks = list(range(0, hi + 1, 5 if hi <= 30 else 10))
        html = html.replace("ys = lin(0, 21, H-m.b, m.t);", f"ys = lin(0, {hi + 1}, H-m.b, m.t);")
        html = html.replace('frame(m, xs, [0,1,2,3], ys, [0,5,10,15,20], false, true, "convection scaling (α)"',
                            f'frame(m, xs, [0,1,2,3], ys, {ticks}, false, true, "convection scaling (α)"')
        html = html.replace("for (const t of [0,5,10,15,20]) s +=", f"for (const t of {ticks}) s +=")

    reps = [
        ('<span>Effect of convection ablation across storms</span>', '<span>Effect of low-level spin ablation across storms</span>'),
        ('<span>Convection dose–response</span>', '<span>Low-level spin dose–response</span>'),
        ('<text class="note" x="${lx+9}" y="${ly+20*FS}">convection</text>', '<text class="note" x="${lx+9}" y="${ly+20*FS}">low-level spin (f3316)</text>'),
        ('aria-label="Deepening removed per storm, convection vs matched control"', 'aria-label="Deepening removed per storm, low-level spin feature 3316 vs matched control"'),
        ('"convection scaling (α)"', '"spin scaling (α)"'),
        ('aria-label="MSLP error as a function of convection scaling for Ida and Haishen"', 'aria-label="MSLP error as a function of low-level spin scaling for Ida and Haishen"'),
        # colours follow the group after the swap: spin keeps glass-green, convection keeps blue
        ('const cols = [C.blue, C.glass, C.grey];', 'const cols = [C.glass, C.blue, C.grey];'),
        ('<title>GraphCast Causal Claims</title>', '<title>GraphCast Causal Claims — Figure 2.5</title>'),
        # both best points sit at x3 for the spin feature: Ida's label goes above-left of its endpoint
        # (clearing Haishen's line), Haishen's goes under its own flat curve near x1
        ('if (st==="ida2021") s += `<text class="note" x="${xs(x[j])+5}" y="${ys(y[j])+16*FS}" fill="${col}">${f(g.baseline)} → ${f(y[j])}</text>`;',
         'if (st==="ida2021") s += `<text class="note" x="${xs(x[j])-4}" y="${ys(y[j])-22*FS}" fill="${col}" text-anchor="end">${f(g.baseline)} → ${f(y[j])}</text>`;'),
        ('else s += `<text class="note" x="${xs(x[j])-5}" y="${ys(y[j])+16*FS}" fill="${col}" text-anchor="end">${f(g.baseline)} → ${f(y[j])}</text>`;',
         'else s += `<text class="note" x="${xs(1)+2}" y="${ys(g.baseline)+15*FS}" fill="${col}">${f(g.baseline)} → ${f(y[j])}</text>`;'),
    ]
    for a, b in reps:
        assert html.count(a) == 1, f"expected exactly one occurrence of: {a[:60]}"
        html = html.replace(a, b)
    DST.write_text(html)
    print("wrote", DST.relative_to(ROOT))
    print("  (a) 3316 per storm:", {s: round(v, 2) for s, v in zip(STORMS, conv)})
    print("  (b) gain:", {s: [round(x, 2) for x in g["err"]] + ["base %.2f" % g["baseline"]] for s, g in f2["gain"].items()})
    print("  (c) order:", [g["label"] for g in f2["groups"]])
    if not render:
        return
    tmp = HERE / "_fig2p5_raw.pdf"
    subprocess.run([str(CHROME), "--headless=new", "--no-sandbox", "--disable-gpu",
                    "--run-all-compositor-stages-before-draw", "--virtual-time-budget=8000",
                    "--no-pdf-header-footer", f"--print-to-pdf={tmp}", f"file://{DST}"],
                   check=True, capture_output=True, timeout=180)
    subprocess.run(["pdfcrop", "--margins", "0", str(tmp), str(PDF)], check=True, capture_output=True)
    tmp.unlink()
    shutil.copy(PDF, ROOT / "paper_clean/images" / PDF.name)
    print("wrote", PDF.relative_to(ROOT), "and paper_clean/images/" + PDF.name)


if __name__ == "__main__":
    main(render="--no-pdf" not in sys.argv)
