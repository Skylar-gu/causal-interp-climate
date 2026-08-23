"""Double-blind release: no author identities, accounts, or assistant traces in tracked files.

`docs/` (verbatim historical documents, provenance headers) and `savar/` (own tests) are skipped.
"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    re.compile(r"skylar", re.I),
    re.compile(r"@gmail\.com", re.I),
    re.compile(r"ec2-user"),
    re.compile(r"\bclaude\b", re.I),
    re.compile(r"anthropic", re.I),
    re.compile(r"claude\.ai/code"),
    re.compile(r"github\.com/(?!google-deepmind|xtibau|paperswithcode|ANONYMIZED)[A-Za-z0-9_-]+"),
]
SKIP_PREFIXES = ("docs/", "savar/")
BINARY = {".npy", ".npz", ".pdf", ".png", ".pt"}


def _tracked():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    return [f for f in out.decode().split("\0") if f]


def test_no_identities_in_tracked_files():
    me = Path(__file__).resolve()
    hits = []
    for rel in _tracked():
        if rel.startswith(SKIP_PREFIXES) or Path(rel).suffix in BINARY:
            continue
        p = ROOT / rel
        if not p.is_file() or p.resolve() == me:
            continue
        text = p.read_text(errors="replace")
        for pat in PATTERNS:
            for i, line in enumerate(text.splitlines(), 1):
                if pat.search(line):
                    hits.append(f"{rel}:{i}: {pat.pattern}  {line.strip()[:80]}")
    assert not hits, "identifying strings found:\n" + "\n".join(hits)
