"""graphcast_sae.paths resolves inside the checkout and honours GC_SCRATCH."""
import importlib
import os
from pathlib import Path


def test_repo_root_and_shipped_files():
    from graphcast_sae import paths
    assert (paths.REPO_ROOT / "README.md").exists()
    assert paths.SAE_WEIGHTS.exists() and paths.SAE_CONFIG.exists()
    assert paths.MESH_GEOM.exists()
    assert paths.RESULTS.is_dir() and paths.DATA.is_dir() and paths.DOCS.is_dir()


def test_default_scratch_is_inside_repo():
    from graphcast_sae import paths
    if "GC_SCRATCH" not in os.environ:
        assert paths.SCRATCH == paths.REPO_ROOT / "scratch"
    assert paths.IID_DUMP.parent == paths.SCRATCH


def test_gc_scratch_env_is_honoured(tmp_path, monkeypatch):
    from graphcast_sae import paths
    monkeypatch.setenv("GC_SCRATCH", str(tmp_path / "elsewhere"))
    try:
        importlib.reload(paths)
        assert paths.SCRATCH == Path(tmp_path / "elsewhere")
        assert paths.IID_DUMP == Path(tmp_path / "elsewhere") / "fs_iid_dump.npy"
        assert paths.MESH_GEOM == paths.DATA / "mesh_2to6_geom.npy"   # shipped, never in scratch
    finally:
        monkeypatch.delenv("GC_SCRATCH")
        importlib.reload(paths)
