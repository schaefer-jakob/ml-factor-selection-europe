"""Tests of the selection methods, stability selection and overlap measures
(Chapter 5)."""
import itertools

import numpy as np
import pandas as pd

from helpers import requires_data
from eufactors import baseline, pca, rolling
from eufactors import selection as sel


def synthetic(n_assets=60, T=380, n_factors=40, k_true=3, prem=0.006, seed=1):
    """Returns from a known model with k_true priced factors and unpriced
    decoys."""
    rng = np.random.default_rng(seed)
    F = rng.normal(0, 0.04, (T, n_factors))
    names = ['f%02d' % i for i in range(n_factors)]
    true = names[:k_true]
    B = rng.normal(0, 1.0, (n_assets, k_true))
    mkt = rng.normal(0.005, 0.045, T)
    bm = rng.normal(1.0, 0.15, n_assets)
    R = (np.outer(mkt, bm).T
         + B @ (F[:, :k_true] + prem).T
         + rng.normal(0, 0.02, (n_assets, T))).T
    idx = pd.period_range('1994-01', periods=T, freq='M')
    return (pd.DataFrame(R, index=idx), mkt,
            pd.DataFrame(F, index=idx, columns=names), true)


def test_fmb_kernel():
    rng = np.random.default_rng(0)
    T, N, K = 200, 40, 3
    F = rng.normal(0, 0.04, (T, K))
    B = rng.normal(1, 0.4, (N, K))
    R = F @ B.T + rng.normal(0, 0.03, (T, N))
    m, t = sel.fmb(R, F)
    assert len(m) == K + 1
    Bhat = sel.betas(R, F)
    se = 0.03 / (0.04 * np.sqrt(T))
    assert np.abs(Bhat - B).max() < 4 * se
    assert np.corrcoef(Bhat.ravel(), B.ravel())[0, 1] > 0.99
    assert np.abs(t[1:]).max() < 3.0


def test_recovery():
    """Every selector finds at least two of three planted factors."""
    R, mkt, Z, true = synthetic()
    for name, fn in sel.selectors().items():
        picked, _ = fn(R, mkt, Z)
        assert len(set(picked) & set(true)) >= 2, (name, picked)
        assert len(picked) <= sel.CAP, (name, picked)


def test_no_false_structure():
    """With no priced candidate, no frequency reaches 0.60."""
    rng = np.random.default_rng(7)
    T, N, P = 380, 60, 40
    idx = pd.period_range('1994-01', periods=T, freq='M')
    Z = pd.DataFrame(rng.normal(0, 0.04, (T, P)), index=idx,
                     columns=['f%02d' % i for i in range(P)])
    mkt = rng.normal(0.005, 0.045, T)
    R = pd.DataFrame(np.outer(mkt, rng.normal(1, .15, N)) + rng.normal(0, .02, (T, N)),
                     index=idx)
    freq, _ = sel.stability_selection(sel.selectors()['LASSO'], R, mkt, Z, n_draws=40)
    assert freq.max() < 0.60, freq.max()


def test_coordinate_descent():
    rng = np.random.default_rng(3)
    n, p = 80, 30
    X = rng.normal(0, 1, (n, p))
    b = np.zeros(p)
    b[[2, 7, 15]] = [1.5, -2.0, 1.0]
    y = X @ b + rng.normal(0, 0.3, n)
    Xs = (X - X.mean(0)) / X.std(0)
    ys = (y - y.mean()) / y.std(ddof=1)
    bh = sel.coord_descent(Xs, ys, 0.02, 1.0)
    assert set(np.argsort(-np.abs(bh))[:3]) == {2, 7, 15}
    assert np.corrcoef(sel.coord_descent(Xs, ys, 1e-9, 1.0),
                       np.linalg.lstsq(Xs, ys, rcond=None)[0])[0, 1] > 0.999
    assert np.all(sel.coord_descent(Xs, ys, 10.0, 1.0) == 0)
    # the L2 term must matter on correlated columns
    Xc = np.column_stack([X[:, 0], X[:, 0] + rng.normal(0, .05, n), X[:, 5:]])
    Xc = (Xc - Xc.mean(0)) / Xc.std(0)
    assert not np.allclose(sel.coord_descent(Xc, ys, 0.05, 1.0),
                           sel.coord_descent(Xc, ys, 0.05, 0.5))


def test_cv_selector_and_clusters():
    R, mkt, Z, true = synthetic(n_assets=120, T=300, n_factors=30, k_true=3)
    R.columns = ['c%02d_pf%d' % (i // 3, i % 3 + 1) for i in range(R.shape[1])]
    pick, capped = sel.penalised_select_cv(R, mkt, Z, alpha=1.0)
    assert set(true) <= set(pick), pick
    assert len(pick) < Z.shape[1] // 2
    assert capped is None
    folds = sel._asset_groups(R)
    assert all(len(set(folds[3 * j:3 * j + 3])) == 1 for j in range(R.shape[1] // 3))
    rng = np.random.default_rng(4)
    base = rng.normal(size=(300, 4))
    Zc = pd.DataFrame(np.column_stack([base, base[:, :2] + 0.01 * rng.normal(size=(300, 2))]),
                      columns=['a', 'b', 'c', 'd', 'a2', 'b2'])
    cm = sel.factor_clusters(Zc)
    assert cm['a'] == cm['a2'] and cm['b'] == cm['b2']
    assert len({cm['a'], cm['b'], cm['c'], cm['d']}) == 4


def test_overlap_measures():
    rng = np.random.default_rng(0)
    assert sel.jaccard({'a', 'b'}, {'a', 'b'}) == 1
    assert sel.jaccard({'a'}, {'b'}) == 0
    assert np.isnan(sel.jaccard(set(), set()))
    p, k = 61, 8
    sets = [set(rng.choice(p, k, replace=False)) for _ in range(200)]
    sim = np.mean([len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2)])
    assert abs(sim - sel.jaccard_random(k, k, p)) < 0.01
    assert sel.jaccard_random(0, 3, p) == 0.0 and np.isnan(sel.jaccard_random(0, 0, p))
    Q, _ = np.linalg.qr(rng.normal(0, 1, (p, p)))
    assert abs(pca.alignment(Q[:, :k], Q[:, :k]) - 1.0) < 1e-9
    assert abs(pca.alignment(Q[:, :k], Q[:, k:2 * k])) < 1e-9
    assert np.isnan(pca.alignment(Q[:, :0], Q[:, :k]))
    vals = []
    for _ in range(200):
        Q1, _ = np.linalg.qr(rng.normal(0, 1, (p, p)))
        Q2, _ = np.linalg.qr(rng.normal(0, 1, (p, p)))
        vals.append(pca.alignment(Q1[:, :k], Q2[:, :k]))
    assert abs(np.mean(vals) - k / p) < 0.01


def test_meinshausen_buehlmann_bound():
    """Appendix C: with q = 8 and p = 61 the bound is uninformative at
    pi = 0.60 and informative at pi = 0.90 or q = 4."""
    assert abs(sel.mb_bound(8, 0.60, 61) - 5.25) < 0.01
    assert abs(sel.mb_bound(8, 0.90, 61) - 1.31) < 0.01
    assert abs(sel.mb_bound(4, 0.60, 61) - 1.31) < 0.01


def test_window_design():
    """26 windows of 72 months, moved by 12, on 375 to 382 months."""
    for T in (375, 380, 382):
        assert len(rolling.window_starts(T)) == 26
    assert sel.WINDOW - 6 * rolling.STEP == 0


def test_candidate_set():
    requires_data()
    U = sel.shared_universe(('DEU', 'GBR', 'FRA'))
    assert len(U) == 65
    assert not (set(U) & set(baseline.CHARS))
    assert set(rolling.CLASSICS) <= set(U)
    assert set(sel.excluded_candidates()) <= set(U)
