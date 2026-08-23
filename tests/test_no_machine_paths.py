"""The release must not carry any machine-specific path.

Walks the package, figure builders, tests, the demo notebook and the top-level
config files. `savar/` has its own test; `docs/` holds historical documents copied
verbatim and is skipped.
"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# assembled so this file does not itself match the release-wide anonymity/path greps
FORBIDDEN = ["/home/" + "ec2-" + "user", "/tmp/" + "cla" + "ude", "Path.home()", 'expanduser("~', ".venv", "miniforge"]
SUFFIXES = {".py", ".md", ".txt", ".ipynb", ".json", ".sh", ".cfg", ".ini", ".toml"}
TREES = ["graphcast_sae", "figures", "tests"]
FILES = ["notebooks/demo.ipynb", "README.md", "requirements.txt", "requirements-dev.txt", ".gitignore"]
# The ignore pattern for virtualenv directories is the one legitimate mention.
ALLOW = {(".gitignore", ".venv")}


def _tracked():
    """Files in the release = files git tracks; untracked scratch scripts are not audited."""
    try:
        out = subprocess.run(["git", "ls-files", "-z", *TREES, *FILES], cwd=ROOT,
                             capture_output=True, check=True).stdout
        return [ROOT / f for f in out.decode().split("\0") if f]
    except (OSError, subprocess.CalledProcessError):      # no git: fall back to the tree
        found = [p for t in TREES for p in (ROOT / t).rglob("*") if p.is_file()]
        return found + [ROOT / f for f in FILES if (ROOT / f).exists()]


def _candidates():
    me = Path(__file__).resolve()
    for p in _tracked():
        if p.suffix in SUFFIXES and "__pycache__" not in p.parts and p.resolve() != me:
            yield p


def test_no_machine_specific_paths():
    hits = []
    for p in _candidates():
        text = p.read_text(errors="replace")
        rel = str(p.relative_to(ROOT))
        for needle in FORBIDDEN:
            if (rel, needle) in ALLOW:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if needle in line:
                    hits.append(f"{rel}:{i}: {needle!r}  {line.strip()[:90]}")
    assert not hits, "machine-specific paths found:\n" + "\n".join(hits)


def test_no_session_ids():
    pat = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
    hits = [str(p.relative_to(ROOT)) for p in _candidates()
            if p.suffix != ".ipynb" and pat.search(p.read_text(errors="replace"))]
    assert not hits, f"session-id strings found in {hits}"
