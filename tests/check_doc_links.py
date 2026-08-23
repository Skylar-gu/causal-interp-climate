"""Every path-like reference in docs/REPRODUCE.md and the group READMEs must exist (or be marked regenerable)."""
import re, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
docs = [ROOT / "docs/REPRODUCE.md"] + sorted(ROOT.glob("graphcast_sae/*/README.md"))
pat = re.compile(r"`((?:graphcast_sae|figures|results|data|docs|tests|notebooks|savar|candidates|scratch)/[A-Za-z0-9_./\-<>*{}$]+|[A-Za-z0-9_]+\.py)`")
missing = {}; ok = 0
for d in docs:
    text = d.read_text()
    for m in pat.finditer(text):
        ref = m.group(1)
        if any(c in ref for c in "<>*{}$"): continue                       # patterns / placeholders
        # a bare `foo.py` refers to a script in the README's own group
        cand = [ROOT / ref] if "/" in ref else [d.parent / ref, ROOT / "figures" / ref, ROOT / "tests" / ref]
        if any(c.exists() for c in cand): ok += 1; continue
        line = text[:m.start()].count("\n") + 1
        ctx = text.splitlines()[line - 1]
        marked = bool(re.search(r"not (shipped|included)|regenerat|not in git|never in git|\(not shipped", ctx, re.I))
        missing.setdefault(str(d.relative_to(ROOT)), []).append((line, ref, "marked regenerable" if marked else "UNMARKED"))
print(f"{ok} references resolved")
bad = 0
for d, items in missing.items():
    for line, ref, tag in items:
        print(f"  {d}:{line}: {ref}  [{tag}]"); bad += tag == "UNMARKED"
sys.exit(1 if bad else 0)
