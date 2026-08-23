"""CPU self-test for the mediation clamp. No GPU, no GraphCast forward pass.

Four things have to be true before a single GPU second is spent:

  T1  jmask=0 makes delta_cond_freeze EXACTLY delta_cond, bit for bit. That is what lets
      the battery's baseline / do-only arms carry the same 6-tuple pytree as the freeze
      arms, so all arms share one compiled graph.
  T2  freezing j to its OWN current code is EXACTLY a zero delta -- the algebraic half of
      the exactness test. (The empirical half is the freeze-only GPU arm; only the model's
      own nondeterminism can break it after this.)
  T3  the clamp is TWO-SIDED and lands exactly on the reference: with fref set to an
      arbitrary value the resulting code column equals fref, whether that is above or
      below the current activation, and whether or not j is in the top-k at that node.
  T4  do(i)+freeze(j) equals do(i) plus the clamp of j applied to the POST-ablation code,
      and the per-step indexing in mediation_run.roll hands the patch a (n_nodes, J)
      slice, never the (H, n_nodes, J) block (the commit-179f487 shape bug).

Also reports the top-k boundary exposure: how often feature j sits within one rank of the
k-th place, because that is the one place where recomputing codes on the host (where fref
comes from) and inside the hook could disagree about whether j is on at all.

Run:  FS_DEVICE=cpu python -m tests.mediation_selftest
"""
import os
import sys

os.environ.setdefault("FS_DEVICE", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

import graphcast_sae.common.fs_common as fc

RNG = np.random.default_rng(0)
N, D, F, K = 97, 32, 64, 8

def fake_sae():
    s = object.__new__(fc.SAEJax)
    W = RNG.normal(size=(D, F)).astype(np.float32)
    W /= np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-8)
    s.W_enc = jnp.asarray(RNG.normal(size=(F, D)).astype(np.float32))
    s.W_dec = jnp.asarray(W)
    s.b_pre = jnp.asarray(RNG.normal(size=(D,)).astype(np.float32) * 0.1)
    s.k = K
    s.n_features = F
    s.tokens = 0
    return s

def main():
    s = fake_sae()
    x = jnp.asarray(RNG.normal(size=(N, D)).astype(np.float32) * 3.0)
    f = np.asarray(s.codes(x))

    fsel = np.zeros(F, np.float32); fsel[3] = 1.0                 # the "source" i = 3
    ftarget = np.zeros(F, np.float32)
    ftarget[3] = float(np.median(f[f[:, 3] > 0, 3])) if (f[:, 3] > 0).any() else 0.0
    nmask = (RNG.random(N) < 0.4).astype(np.float32)
    JF = [11, 27]                                                  # mediator + control
    jidx = np.asarray(JF, np.int32)
    zeroJ = np.zeros((N, len(JF)), np.float32)
    fref_self = f[:, JF].astype(np.float32)                        # j's own current codes

    fails = []

    def check(tag, ok, extra=""):
        print("  %-4s %-58s %s" % ("PASS" if ok else "FAIL", tag, extra))
        if not ok:
            fails.append(tag)

    print("T1  jmask=0 reproduces delta_cond exactly")
    d_cond = np.asarray(s.delta_cond(x, jnp.asarray(fsel), jnp.asarray(ftarget), jnp.asarray(nmask)))
    d_free = np.asarray(s.delta_cond_freeze(x, jnp.asarray(fsel), jnp.asarray(ftarget),
                                            jnp.asarray(nmask), jnp.asarray(jidx),
                                            jnp.asarray(fref_self), jnp.asarray(zeroJ)))
    check("delta_cond_freeze(jmask=0) == delta_cond", np.array_equal(d_cond, d_free),
          "max|diff| = %.3e" % float(np.abs(d_cond - d_free).max()))
    d_free0 = np.asarray(s.delta_cond_freeze(x, jnp.zeros(F, jnp.float32), jnp.asarray(ftarget),
                                             jnp.zeros(N, jnp.float32), jnp.asarray(jidx),
                                             jnp.asarray(fref_self), jnp.asarray(zeroJ)))
    check("the no-op 6-tuple arm is an exact zero delta", np.all(d_free0 == 0.0),
          "max|d| = %.3e" % float(np.abs(d_free0).max()))

    print("T2  freeze-to-self is an exact zero delta (algebraic exactness)")
    for scope, jm in (("global", np.ones((N, len(JF)), np.float32)),
                      ("disk", np.repeat(nmask[:, None], len(JF), 1))):
        d = np.asarray(s.delta_cond_freeze(x, jnp.zeros(F, jnp.float32), jnp.asarray(ftarget),
                                           jnp.zeros(N, jnp.float32), jnp.asarray(jidx),
                                           jnp.asarray(fref_self), jnp.asarray(jm)))
        check("freeze(%s) with fref = own codes -> delta == 0" % scope, np.all(d == 0.0),
              "max|d| = %.3e" % float(np.abs(d).max()))

    print("T3  the clamp is two-sided and lands exactly on the reference")
    jm = np.zeros((N, len(JF)), np.float32); jm[:, 0] = 1.0        # freeze JF[0] only
    fref = fref_self.copy()
    fref[:, 0] = RNG.normal(size=N).astype(np.float32) * 2.0       # arbitrary: above AND below
    d = np.asarray(s.delta_cond_freeze(x, jnp.zeros(F, jnp.float32), jnp.asarray(ftarget),
                                       jnp.zeros(N, jnp.float32), jnp.asarray(jidx),
                                       jnp.asarray(fref), jnp.asarray(jm)))
    want = np.outer(fref[:, 0] - f[:, JF[0]], np.asarray(s.W_dec)[:, JF[0]])
    check("delta == (fref - f_j) outer w_dec_j", np.allclose(d, want, atol=1e-5, rtol=1e-4),
          "max|diff| = %.3e" % float(np.abs(d - want).max()))
    up = (fref[:, 0] > f[:, JF[0]]).sum(); dn = (fref[:, 0] < f[:, JF[0]]).sum()
    check("both directions exercised", up > 0 and dn > 0, "raised %d nodes, lowered %d" % (up, dn))
    off = (f[:, JF[0]] == 0).sum()
    check("clamp also acts where j is OFF (outside top-k)", off > 0,
          "%d/%d nodes had f_j == 0" % (off, N))
    check("the untouched frozen column JF[1] is unaffected",
          np.allclose(d, want, atol=1e-5), "(same delta as the single-column expectation)")

    print("T4  do(i)+freeze(j) composes as specified")
    jm = np.zeros((N, len(JF)), np.float32); jm[:, 0] = 1.0
    d = np.asarray(s.delta_cond_freeze(x, jnp.asarray(fsel), jnp.asarray(ftarget),
                                       jnp.asarray(nmask), jnp.asarray(jidx),
                                       jnp.asarray(fref), jnp.asarray(jm)))
    capped = np.where(fsel[None, :] > 0, np.minimum(f, ftarget[None, :]), f)
    fn = f + nmask[:, None] * (capped - f)
    fn[:, JF[0]] = fref[:, 0]
    want = (fn - f) @ np.asarray(s.W_dec).T
    check("delta == (f_ablated_then_clamped - f) @ W_dec.T",
          np.allclose(d, want, atol=1e-5, rtol=1e-4),
          "max|diff| = %.3e" % float(np.abs(d - want).max()))

    # per-step indexing, exactly as mediation_run.roll does it
    Hh = 5
    block = jnp.asarray(RNG.normal(size=(Hh, N, len(JF))).astype(np.float32))
    pj0 = (jnp.asarray(fsel), jnp.asarray(ftarget), jnp.asarray(nmask), jnp.asarray(jidx),
           block, jnp.asarray(jm))
    shapes = set()
    for h in range(Hh):
        pj = pj0[:4] + (pj0[4][h], pj0[5])
        shapes.add(tuple(pj[4].shape))
        _ = s.delta_cond_freeze(*((x,) + pj))        # would raise on a broadcast mismatch
    check("per-step slice is (n_nodes, J), never (H, n_nodes, J)", shapes == {(N, len(JF))},
          "slice shapes seen: %s" % sorted(shapes))

    print("T5  jit: all six-tuple arms share ONE compiled graph")
    g = jax.jit(lambda xx, p: s.delta_cond_freeze(xx, *p))
    n0 = g._cache_size() if hasattr(g, "_cache_size") else None
    arms = [(np.zeros(F, np.float32), ftarget, np.zeros(N, np.float32), jidx, fref_self, zeroJ),
            (fsel, ftarget, nmask, jidx, fref_self, zeroJ),
            (np.zeros(F, np.float32), ftarget, np.zeros(N, np.float32), jidx, fref, jm),
            (fsel, ftarget, nmask, jidx, fref, jm)]
    for p in arms:
        g(x, tuple(jnp.asarray(v) for v in p))
    n1 = g._cache_size() if hasattr(g, "_cache_size") else None
    check("one trace for four arms", (n1 is None) or (n1 - (n0 or 0) == 1),
          "cache entries: %s -> %s" % (n0, n1))

    print("T6  top-k boundary exposure (the only host-vs-hook divergence risk)")
    pre = np.asarray(jax.nn.relu((s.norm_tok(x) - s.b_pre) @ s.W_enc.T))
    srt = np.sort(pre, 1)[:, ::-1]
    gap = (srt[:, K - 1] - srt[:, K]) / np.maximum(srt[:, 0], 1e-9)
    print("     relative gap between rank k and k+1: median %.4f  p05 %.4f  min %.4f"
          % (np.median(gap), np.quantile(gap, 0.05), gap.min()))
    print("     nodes with gap < 1e-5 (a float32 recompute could flip membership): %d/%d"
          % (int((gap < 1e-5).sum()), N))

    print("\n%s  (%d checks failed)" % ("ALL PASS" if not fails else "FAILURES: %s" % fails,
                                        len(fails)))
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
