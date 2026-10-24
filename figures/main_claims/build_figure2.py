"""Figure 2 (camera-ready): the submitted layout and look, on min-MSLP trajectory RMSE.

  a  restore-to-normal of the single features 2401 (convection) and 3316 (low-level vorticity),
     every storm, each against its random-feature control: results/skill/mech_{2401,3316}
  b  convection-group gain sweep on Ida and Haishen from the controlled sweep (J1), with the
     in-box control at every gain: results/skill/suff_ctl/rev_verdict.json (`err_mslp`)
  c  feature 3316 gain sweep on Ida and Haishen, with its random-feature control arms at
     alpha = 0 and 3: results/skill/gain_3316

MSLP error is rev_analyze.arm_metrics `err_mslp` (box minimum MSLP against ERA5 over leads up
to max(ERA5 peak, +36 h); forecast step k aligned with analysis k+1), for both b and c.

Derives figure2_final_web_notitle_print.html from figure2_web_notitle_print.html (same CSS and
drawing helpers) by replacing the D.fig2 data block, the three Figure 2 panel scripts and the
panel titles, then prints it with headless chromium through playwright.

    python3 figures/main_claims/build_figure2.py            # html + pdf (needs playwright)
    python3 figures/main_claims/build_figure2.py --no-pdf   # html only
"""
import json, re, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SK = ROOT / "results/skill"
SRC = HERE / "figure2_web_notitle_print.html"
DST = HERE / "figure2_final_web_notitle_print.html"
PDF = HERE / "figure2_interventions_final.pdf"
STORMS = ["ida2021", "michael2018", "haishen2020", "goni2020", "haiyan2013", "patricia2015", "wilma2005"]
SWEPT = ["ida2021", "haishen2020"]


def restore(name):
    m = json.load(open(SK / f"{name}/verdict.json"))["metrics"]
    return ([m[s]["arms"]["conv-normal"]["d_deepen"] for s in STORMS],
            [m[s]["arms"]["rand-normal"]["d_deepen"] for s in STORMS])


def conv_sweep():
    M = json.load(open(SK / "suff_ctl/rev_verdict.json"))["metrics"]
    gains = [0, 0.5, 1.25, 1.5, 2, 2.5, 3]
    out = {}
    for s in SWEPT:
        a = M[s]["arms"]
        out[s] = dict(gains=gains, err=[a["gain-%g" % g]["err_mslp"] for g in gains],
                      ctl=[a["rand-gain-%g" % g]["err_mslp"] for g in gains],
                      baseline=a["baseline"]["err_mslp"])
    return out


def spin_sweep():
    T = np.load(SK / "gain_3316/era5_truth.npy", allow_pickle=True).item()
    gains = [0, 1.25, 1.5, 1.75, 2, 2.5, 3]
    out = {}
    for s in SWEPT:
        res = np.load(SK / f"gain_3316/run_{s}.npy", allow_pickle=True).item()["res"]
        t = np.asarray(T[s]["mslp_min"])

        def rmse(arm):
            mm = np.asarray(res[arm]["mslp_min"]); H = len(mm)
            assert np.isfinite(mm).all(), (s, arm)
            w = np.arange(min(max(int(np.argmin(t)), 6), H))
            return float(np.sqrt(np.mean((mm[w] - t[1:H + 1][w]) ** 2)))
        out[s] = dict(gains=gains, err=[rmse("gain-%g" % g) for g in gains], baseline=rmse("baseline"),
                      ctl={"0": rmse("rand-normal"), "3": rmse("rand-gain-3")})
    return out


conv, cctl = restore("mech_2401")
spin, sctl = restore("mech_3316")
D2 = dict(storms=STORMS, conv=conv, cctl=cctl, spin=spin, sctl=sctl, gconv=conv_sweep(), gspin=spin_sweep())

JS = r"""// ---------- Figure 2a
{
  const nice = {ida2021:"Ida", michael2018:"Michael", haishen2020:"Haishen", goni2020:"Goni", haiyan2013:"Haiyan", patricia2015:"Patricia", wilma2005:"Wilma"};
  const F = D.fig2, n = F.storms.length;
  const idx = F.spin.map((v,i)=>i).sort((i,j)=>F.spin[i]-F.spin[j]);
  const m = {l:64*FS, r:16, t:8, b:38*FS};
  const xs = lin(-.85, 8.15, m.l, W-m.r);
  const ys = k => H-m.b - (k+0.5)*(H-m.t-m.b)/n, dy = 3.1*FS;
  let s = frame(m, xs, [0,2,4,6,8], null, null, true, false, "deepening removed (hPa)");
  s += `<line x1="${xs(0)}" x2="${xs(0)}" y1="${m.t}" y2="${H-m.b}" stroke="var(--grid)" stroke-width="1"/>`;
  idx.forEach((i,k)=>{
    const y = ys(k);
    [[F.conv[i], F.cctl[i], C.blue, -dy], [F.spin[i], F.sctl[i], C.glass, dy]].forEach(([v,c,col,o])=>{
      s += `<line x1="${xs(c)}" x2="${xs(v)}" y1="${y+o}" y2="${y+o}" stroke="var(--pale)" stroke-width="1.1"/>`;
      s += `<circle cx="${xs(c)}" cy="${y+o}" r="3.0" fill="var(--panel)" stroke="${C.grey}" stroke-width="1.1"/>`;
      s += `<circle cx="${xs(v)}" cy="${y+o}" r="3.3" fill="${col}"/>`;
    });
    s += `<text class="ylab" x="${m.l-8}" y="${y+4*FS}" text-anchor="end">${nice[F.storms[i]]}</text>`;
  });
  document.getElementById("f2a").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Deepening removed per storm, features 2401 and 3316 vs random controls">${s}</svg>`;
}
// ---------- Figure 2b, 2c (same axes)
function doseResponse(id, key, xlabel, aria, side){
  const m = {l:52*FS, r:16, t:8, b:38*FS};
  const xs = lin(-.12, 3.12, m.l, W-m.r), ys = lin(0, 21, H-m.b, m.t);
  let s = frame(m, xs, [0,1,2,3], ys, [0,5,10,15,20], false, true, xlabel, "MSLP error (hPa)", v => v===0 ? "0" : "×"+v);
  for (const t of [0,5,10,15,20]) s += `<text class="tick" x="${m.l-6}" y="${ys(t)+4*FS}" text-anchor="end">${t}</text>`;
  s += `<line x1="${xs(1)}" x2="${xs(1)}" y1="${m.t}" y2="${H-m.b}" stroke="var(--grid)" stroke-width="1"/>`;
  const mark = (x,y,r,col,sq,fill) => sq ? `<rect x="${x-r}" y="${y-r}" width="${2*r}" height="${2*r}" fill="${fill}" stroke="${col}" stroke-width="1.2"/>`
                                          : `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="${col}" stroke-width="1.2"/>`;
  const series = [["ida2021","Ida",C.blue,false],["haishen2020","Haishen",C.glass,true]];
  series.forEach(([st,lab,col,sq])=>{
    const g = D.fig2[key][st];
    const pts = [[1, g.baseline], ...g.gains.map((v,i)=>[v, g.err[i]]).filter(p=>p[0]>0)].sort((p,q)=>p[0]-q[0]);
    const x = pts.map(p=>p[0]), y = pts.map(p=>p[1]);
    if (Array.isArray(g.ctl)) {             // in-box control at every gain
      const cp = [[1, g.baseline], ...g.gains.map((v,i)=>[v, g.ctl[i]]).filter(p=>p[0]>0)].sort((p,q)=>p[0]-q[0]);
      const yc = cp.map(p=>p[1]);
      s += `<polyline fill="none" stroke="${C.grey}" stroke-width="1" stroke-dasharray="3 2" points="${x.map((v,i)=>xs(v)+","+ys(yc[i])).join(" ")}"/>`;
    } else {                                // random-feature control at the end gains
      for (const [gk, v] of Object.entries(g.ctl)) s += mark(xs(+gk), ys(v), 3.0, C.grey, sq, "var(--panel)");
    }
    s += `<polyline fill="none" stroke="${col}" stroke-width="1.6" points="${x.map((v,i)=>xs(v)+","+ys(y[i])).join(" ")}"/>`;
    x.forEach((v,i)=> s += mark(xs(v), ys(y[i]), 2.8, col, sq, col));
    s += mark(xs(0), ys(g.err[0]), 3.2, col, sq, "var(--panel)");
    let j = 0; y.forEach((v,i)=>{ if (v < y[j]) j = i; });
    s += mark(xs(x[j]), ys(y[j]), 4.4, col, sq, col);
    const ly = st==="ida2021" ? m.t+12*FS : m.t+27*FS;
    const lx = side==="left" ? m.l+8 : W-m.r, anchor = side==="left" ? "start" : "end";
    s += `<text class="note arrowfont" x="${lx}" y="${ly}" fill="${col}" text-anchor="${anchor}">${lab} ${f(g.baseline)} &#8594; ${f(y[j])}</text>`;
  });
  document.getElementById(id).innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${aria}">${s}</svg>`;
}
doseResponse("f2b", "gconv", "convection scaling (α)", "MSLP error against convection scaling for Ida and Haishen", "left");
doseResponse("f2c", "gspin", "vorticity scaling (α)", "MSLP error against vorticity scaling for Ida and Haishen", "right");
"""

LEGEND = """  <div class="f2legend">
    <span><i class="dot" style="background:var(--blue)"></i>feature 2401 (convection; b: group 2401/2067/3174)</span>
    <span><i class="dot" style="background:var(--glass)"></i>feature 3316 (vorticity)</span>
    <span><i class="dot ring"></i>random-feature control</span>
    <span><i class="dash"></i>in-box control (b)</span>
    <span class="sub">b, c: <i class="dot ring small"></i>0: storm excess removed &nbsp;|&nbsp; &times;1: unmodified forecast &nbsp;|&nbsp; <i class="dot big"></i>large marker: lowest error</span>
  </div>
"""
LEGEND_CSS = """
.f2legend{display:flex;flex-wrap:wrap;justify-content:center;gap:4px 22px;margin-top:6px;font-size:17px;color:var(--muted)}
.f2legend span{display:inline-flex;align-items:center;gap:7px}
.f2legend .sub{flex-basis:100%;justify-content:center;font-size:15px}
.f2legend .dot{display:inline-block;width:10px;height:10px;border-radius:50%}
.f2legend .ring{background:var(--panel);border:1.4px solid var(--grey)}
.f2legend .small{width:8px;height:8px;border-color:var(--muted)} .f2legend .big{width:12px;height:12px;background:var(--muted)}
.arrowfont{font-family:"IBM Plex Sans","Noto Sans Math",sans-serif}
.f2legend .dash{display:inline-block;width:18px;border-top:1.6px dashed var(--grey)}
"""


def build():
    s = SRC.read_text()
    m = re.search(r'const D = (\{.*?\});\n', s, re.S)
    D = json.loads(m.group(1)); D["fig2"] = D2
    s = s[:m.start(1)] + json.dumps(D) + s[m.end(1):]
    i = s.index("// ---------- Figure 2a"); j = s.index("// ----------", s.index("// ---------- Figure 2c") + 10)
    s = s[:i] + JS + s[j:]
    for old, new in [("Effect of convection ablation across storms", "Convection and vorticity restored to storm-free normal"),
                     ("Effects across calibrated feature groups", "Low-level vorticity dose–response")]:
        assert s.count(old) == 1, old; s = s.replace(old, new)
    a = s.index('<div id="f2c"></div></div>\n  </div>\n') + len('<div id="f2c"></div></div>\n  </div>\n')
    s = s[:a] + LEGEND + s[a:]
    s = s.replace("</style>", LEGEND_CSS + "</style>", 1)
    s = s.replace("family=IBM+Plex+Sans:", "family=Noto+Sans+Math&family=IBM+Plex+Sans:", 1)
    DST.write_text(s)
    print("->", DST)


def render():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=2)
        pg.goto(DST.resolve().as_uri()); pg.wait_for_timeout(1500)
        el = pg.locator("figure:not(.hide)").first
        bb = el.bounding_box()
        el.screenshot(path=str(PDF.with_suffix(".png")))
        # print only the figure: hide everything else, size the page to the figure box
        pg.add_style_tag(content="@page{size:%dpx %dpx;margin:0} .wrap>*:not(figure:not(.hide)){display:none}"
                         % (bb["width"] + 24, bb["height"] + 24))
        pg.pdf(path=str(PDF), print_background=True, prefer_css_page_size=True)
        b.close()
    print("->", PDF)


if __name__ == "__main__":
    build()
    if "--no-pdf" not in sys.argv:
        render()
