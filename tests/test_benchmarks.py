"""Tests of the SDF shrinkage, the lucky-factors bootstrap and the random
dense directions."""
import numpy as np
import pandas as pd

from eufactors import dense_benchmark as db
from eufactors import kns
from eufactors import lucky_factors as lf


def _panel(T=400, N=90, p=30, planted=(0, 1), prem=0.006, seed=7):
    """Test assets driven by the market and two priced candidates; the other
    candidates are unpriced decoys with the same volatility."""
    rng = np.random.default_rng(seed)
    m = rng.normal(0.005, 0.045, T)
    Z = rng.normal(0.0, 0.03, (T, p))
    for j in planted:
        Z[:, j] += prem
    B = rng.normal(0, 1, (N, len(planted)))
    R = (np.outer(m, rng.normal(1, 0.2, N)) + Z[:, list(planted)] @ B.T
         + rng.normal(0, 0.02, (T, N)))
    return R, m, Z


def test_kns():
    rng = np.random.default_rng(1)
    F = rng.normal(0.002, 0.03, (300, 12))
    mu, S = kns.moments(F)
    g = kns.gamma_from_kappa(0.5, S, 300)
    assert np.allclose(kns.l2_sdf(mu, S, g), kns.dual_penalty(mu, S, g, 0.0), atol=1e-8)
    path = kns.sparsity_path(mu, S, g, ks=(1, 3, 5))
    assert set(path) == {1, 3, 5} and all(path[k][1] >= k for k in path)
    assert path[1][1] == 1
    assert abs(kns.r2_oos(mu, S, np.linalg.solve(S, mu)) - 1) < 1e-10
    surf_raw, _ = kns.cv_surface(F, kappas=np.array([0.3]), ks=(1,))
    surf_pc, _ = kns.cv_surface(F, kappas=np.array([0.3]), ks=(1,), rotate_pc=True)
    assert abs(surf_raw[0, 0] - surf_pc[0, 0]) < 1e-8
    Fa, _, _ = kns.prepare(F[:, 1:], F[:, 0])
    assert np.abs(np.corrcoef(np.column_stack([F[:, 0], Fa]).T)[0, 1:]).max() < 1e-10


def test_lucky_factors():
    R, m, Z = _panel()
    A_plus, a, s, d0 = lf.augmented_intercepts(R, m[:, None], Z)
    X = np.column_stack([np.ones(len(m)), m, Z[:, 3]])
    a3 = np.linalg.lstsq(X, R, rcond=None)[0][0]
    assert np.allclose(A_plus[3], a3, atol=1e-12)
    A0, a0, s0, _ = lf.augmented_intercepts(R, m[:, None], Z - d0[None, :])
    assert np.allclose(A0, a0[None, :], atol=1e-12)
    assert np.abs(lf.si(A0, a0, s0)).max() < 1e-12
    chosen, steps = lf.select(R, m, Z, n_boot=300)
    assert set(chosen) >= {'f0', 'f1'}, chosen
    assert len(set(chosen) - {'f0', 'f1'}) <= 1
    assert not steps[-1]['entered']
    Rn = np.outer(m, np.ones(R.shape[1])) + np.random.default_rng(3).normal(0, 0.02, R.shape)
    chosen0, _ = lf.select(Rn, m, Z, n_boot=300)
    assert len(chosen0) <= 1, chosen0


def test_random_directions():
    rng = np.random.default_rng(2)
    V = db.random_directions(40, 8, rng)
    assert np.allclose(V.T @ V, np.eye(8), atol=1e-12)
    Z = pd.DataFrame(rng.normal(size=(100, 40)))
    assert db.project(Z, Z, V).shape == (100, 8)
