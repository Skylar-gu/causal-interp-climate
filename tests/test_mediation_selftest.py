"""CPU self-test of the mediation clamp (T1-T4 in tests/mediation_selftest.py).

Needs the JAX environment (graphcast + jax); skipped elsewhere. ~3 s on CPU.
"""
import pytest

pytest.importorskip("jax")
pytest.importorskip("graphcast")


def test_mediation_clamp_selftest(monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    from tests import mediation_selftest as m
    assert m.main() == 0
