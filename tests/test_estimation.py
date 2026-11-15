"""Tests of the Fama-MacBeth engine, the GRS and KRS statistics, power and the
baseline pipeline (Chapter 4)."""
import numpy as np
import pandas as pd

from helpers import requires_data
from eufactors import baseline, data, fmb, krs, power
from eufactors import selection as sel

FACTORS = ['F1', 'F2', 'F3']
EF_TRUE = np.array([0.0050, 0.0030, 0.0010])
SD_F, SD_EPS = 0.040, 0.020
WINDOW, MIN_PERIODS, MIN_STOCKS = 36, 24, 40


def _raises(fn):
    try:
        fn()
        return False
    except Exception:
        return True


def make_panel(seed=11, n=60, t=400):
    """Synthetic panel with constant betas and zero alpha, so lambda = E[f]
    is the truth. Factors are i.i.d. across months."""
    rng = np.random.default_rng(seed)
    k = len(FACTORS)
    dates = pd.date_range('1990-01-31', periods=t, freq='ME')
    F = rng.normal(0.0, SD_F, (t, k)) + EF_TRUE
    B = rng.normal(1.0, 0.5, (n, k))
    R = F @ B.T + rng.normal(0.0, SD_EPS, (t, n))
    panel = pd.DataFrame({
        'gvkey': np.repeat(['%04d' % i for i in range(n)], t),
        'iid': '01',
        'datadate': np.tile(dates, n),
        'ret_usd_excess': R.T.reshape(-1),
    })
    for j, c in enumerate(FACTORS):
        panel[c] = np.tile(F[:, j], n)
    return panel, pd.DataFrame(F, index=dates, columns=FACTORS), B


def _pooled_slope(lam, fp):
    """Pooled through-the-origin slope of lambda_t on f_t."""
    common = lam.index.intersection(fp.index)
    L = lam.loc[common, ['lambda_' + c for c in FACTORS]].to_numpy().ravel()
    f = fp.loc[common, FACTORS].to_numpy().ravel()
    ok = ~np.isnan(L)
    return float((L[ok] @ f[ok]) / (f[ok] @ f[ok]))


# ------------------------------------------------------------ rolling betas ---
def test_rolling_engine_recovers_premia():
    """Zero-alpha control for the rolling two-pass engine. The attenuation
    slope matches errors-in-variables theory, 0.25 / (0.25 + 0.0004 /
    (36 * 0.0016)) = 0.973, and a one-month look-ahead pushes it to one."""
    panel, fp, B = make_panel()
    betas = fmb.rolling_betas(panel, FACTORS, window=WINDOW, min_periods=MIN_PERIODS)
    assert len(betas) == 60 * (400 - MIN_PERIODS)
    assert int(betas['n_obs'].max()) == WINDOW
    assert int(betas['n_obs'].min()) == MIN_PERIODS
    lam = fmb.cross_sectional_lambdas(betas, panel, FACTORS, min_stocks=MIN_STOCKS)
    common = lam.index.intersection(fp.index)
    err = np.abs(lam.loc[common, ['lambda_' + c for c in FACTORS]].mean().to_numpy()
                 - fp.loc[common].mean().to_numpy()).max()
    assert len(lam) > 350
    assert err < 0.0015, err
    slope = _pooled_slope(lam, fp)
    assert abs(slope - 0.9730) < 0.02, slope
    ahead = betas.copy()
    ahead['datadate'] = ahead['datadate'] + pd.offsets.MonthEnd(-1)
    lam_ahead = fmb.cross_sectional_lambdas(ahead, panel, FACTORS, min_stocks=MIN_STOCKS)
    slope_ahead = _pooled_slope(lam_ahead, fp)
    assert slope_ahead - slope > 0.015, (slope, slope_ahead)


def _corrupt(panel, mask):
    p = panel.copy()
    rng = np.random.default_rng(999)
    k = mask.sum()
    p.loc[mask, 'ret_usd_excess'] = rng.normal(0, 1.0, k)
    for c in FACTORS:
        p.loc[mask, c] = rng.normal(0, 1.0, k)
    return p


def _betas_indexed(panel):
    b = fmb.rolling_betas(panel, FACTORS, window=WINDOW, min_periods=MIN_PERIODS)
    return b.set_index(['gvkey', 'iid', 'datadate']).sort_index()


def _max_delta(a, b, stamps):
    cols = ['beta_' + c for c in FACTORS]
    ia = a[a.index.get_level_values('datadate').isin(stamps)]
    ib = b.loc[ia.index]
    return float(np.abs(ia[cols].to_numpy() - ib[cols].to_numpy()).max())


def test_rolling_betas_strict_causality():
    """The beta used for month t depends on months t - window to t - 1 only."""
    panel, _, _ = make_panel(t=240)
    dates = np.sort(panel['datadate'].unique())
    t_star = dates[120]
    clean = _betas_indexed(panel)
    future = _betas_indexed(_corrupt(panel, panel['datadate'] >= t_star))
    assert _max_delta(clean, future, dates[:121]) == 0.0
    assert _max_delta(clean, future, dates[121:122]) > 0.1
    previous = _betas_indexed(_corrupt(panel, panel['datadate'] == dates[119]))
    assert _max_delta(clean, previous, dates[120:121]) > 0.1
    assert _max_delta(clean, previous, dates[119:120]) == 0.0


def test_cross_section_against_hand_ols():
    rng = np.random.default_rng(4)
    N, T, K = 40, 24, 3
    fc = ['A', 'B', 'C']
    rows, bet = [], {}
    dates = pd.date_range('2000-01-31', periods=T, freq='ME')
    for i in range(N):
        b = rng.normal(1, .4, K)
        bet[i] = b
        for d in dates:
            rows.append(dict(gvkey=str(i), iid='01', datadate=d,
                             ret_usd_excess=float(rng.normal(.01, .05)),
                             **{f: 0.0 for f in fc}))
    panel = pd.DataFrame(rows)
    betas = pd.DataFrame([dict(gvkey=str(i), iid='01', datadate=d,
                               **{'beta_' + f: bet[i][j] for j, f in enumerate(fc)})
                          for i in range(N) for d in dates])
    lam = fmb.cross_sectional_lambdas(betas, panel, fc, min_stocks=10)
    sub = panel[panel.datadate == dates[0]].sort_values('gvkey')
    X = np.column_stack([np.ones(N)] + [[bet[int(g)][j] for g in sub.gvkey]
                                        for j in range(K)])
    hand = np.linalg.lstsq(X, sub.ret_usd_excess.to_numpy(), rcond=None)[0]
    got = lam.loc[lam.index[0], ['gamma0'] + ['lambda_' + f for f in fc]].to_numpy()
    assert np.abs(hand - got).max() < 1e-10


def test_gamma0_restricted_and_power():
    rng = np.random.default_rng(8)
    T, N = 150, 40
    months = pd.date_range('2000-01-31', periods=T, freq='ME')
    beta = rng.normal(1, 0.5, N)
    rows = []
    for i in range(N):
        r = 0.006 * beta[i] + rng.normal(0, 0.01, T)
        rows.append(pd.DataFrame({'gvkey': 'a%02d' % i, 'iid': '01', 'datadate': months,
                                  'ret_usd_excess': r, 'beta_F': beta[i]}))
    panel = pd.concat(rows, ignore_index=True)
    lam = fmb.cross_sectional_lambdas(panel[['gvkey', 'iid', 'datadate', 'beta_F']],
                                      panel, ['F'], min_stocks=10, intercept=False)
    assert (lam['gamma0'] == 0).all()
    assert abs(lam['lambda_F'].mean() - 0.006) < 0.001
    z = power.Z_ALPHA
    assert abs(power._power(0.0, z) - 0.05) < 1e-6
    assert abs(power._power(z + power.Z_BETA, z) - 0.80) < 1e-3


# ---------------------------------------------------------- standard errors ---
def test_newey_west_against_ar1():
    """Newey-West approaches the long-run standard error of an AR(1)."""
    rng = np.random.default_rng(9)
    rho, T = 0.6, 40000
    e = rng.normal(0, 1, T)
    x = np.empty(T)
    x[0] = e[0]
    for t in range(1, T):
        x[t] = rho * x[t - 1] + e[t]
    lrv = 1.0 / (1 - rho ** 2) * (1 + rho) / (1 - rho)
    analytic = np.sqrt(lrv / T)
    nw = fmb.newey_west_se(x, lags=60)
    assert nw > x.std(ddof=1) / np.sqrt(T)
    assert abs(nw / analytic - 1) < 0.10


def test_newey_west_cov():
    rng = np.random.default_rng(3)
    x = np.convolve(rng.normal(size=400), [1, 0.6, 0.3], mode='same')
    assert abs(fmb.newey_west_cov(x[:, None])[0, 0] - fmb.newey_west_se(x) ** 2) < 1e-15
    V = fmb.newey_west_cov(rng.normal(size=(300, 3)))
    assert np.allclose(V, V.T) and np.linalg.eigvalsh(V).min() > -1e-15
    assert fmb.newey_west_lags(344) == 5


def test_chi2_sf():
    assert abs(fmb.chi2_sf(3.841458820694124, 1) - 0.05) < 1e-9
    assert abs(fmb.chi2_sf(12.591587243743977, 6) - 0.05) < 1e-9


def test_krs_influence_against_finite_differences():
    """The analytic influence function equals the empirical one, on a
    misspecified data-generating process."""
    rng = np.random.default_rng(11)
    T, N, K = 240, 25, 3
    F = rng.normal(0.004, 0.04, (T, K))
    B = rng.normal(1, 0.5, (N, K))
    alpha = rng.normal(0, 0.002, N)
    R = alpha + F @ B.T + rng.normal(0, 0.03, (T, N))
    psi, gamma, _ = krs.influence(R, F)
    g0, _ = krs._estimate(R, F)
    assert np.allclose(g0, gamma)
    h, worst = 1e-6, 0.0
    for t in (0, 57, 199):
        w = np.full(T, (1 - h) / T)
        w[t] += h
        g_h, _ = krs._estimate(R, F, w)
        fd = (g_h - gamma) / h
        worst = max(worst, np.abs(fd - psi[t]).max() / (np.abs(psi[t]).max() + 1e-12))
    assert worst < 1e-3, worst
    assert np.abs(psi.mean(0)).max() < 1e-10


def test_krs_reduces_to_shanken():
    """Under correct specification the KRS variance of lambda equals
    Shanken's (1 + c) Omega + Sigma_f (equation B1)."""
    rng = np.random.default_rng(5)
    T, N, K = 3000, 30, 2
    F = rng.normal(0.006, 0.04, (T, K))
    B = rng.normal(1, 0.5, (N, K))
    R = F @ B.T + rng.normal(0, 0.03, (T, N))
    out = krs.krs_standard_errors(R, F, nw_lags=0)
    Rc, Fc = R - R.mean(0), F - F.mean(0)
    Sf = Fc.T @ Fc / T
    Bh = Rc.T @ Fc / T @ np.linalg.inv(Sf)
    lam = out['gamma'][1:]
    c = lam @ np.linalg.solve(Sf, lam)
    X = np.column_stack([np.ones(N), Bh])
    H = np.linalg.inv(X.T @ X)
    E = Rc - Fc @ Bh.T
    Om = H @ X.T @ (E.T @ E / T) @ X @ H
    V_sh = ((1 + c) * Om[1:, 1:] + Sf) / T
    ratio = out['se_gamma'][1:] / np.sqrt(np.diag(V_sh))
    assert np.abs(ratio - 1).max() < 0.05, ratio


# --------------------------------------------------------------------- GRS ---
def test_grs():
    rng = np.random.default_rng(7)
    ps = []
    for _ in range(200):
        T, N, K = 300, 15, 3
        F = rng.normal(0, 0.04, (T, K))
        B = rng.normal(1, 0.3, (N, K))
        R = F @ B.T + rng.normal(0, 0.05, (T, N))
        ps.append(baseline.grs_test(R, F)['p_value'])
    ps = np.array(ps)
    assert 0.42 < ps.mean() < 0.58
    assert 0.01 < (ps < 0.05).mean() < 0.11
    assert baseline.grs_test(R + 0.005, F)['p_value'] < 0.01
    assert _raises(lambda: baseline.grs_test(np.zeros((10, 20)), np.zeros((10, 2))))
    assert np.isnan(baseline.grs_or_nan(rng.normal(size=(10, 20)), rng.normal(size=(10, 2)))['grs'])
    assert abs(baseline.f_sf(4.0, 2, 10) - 0.052922) < 1e-5
    assert abs(baseline.f_sf(1.0, 1, 1) - 0.5) < 1e-6


def test_min_assets():
    assert baseline.min_assets(60, 6) == 45
    assert baseline.min_assets(15, 6) == 11
    assert baseline.min_assets(3, 6) == 3


def test_synthetic_positive_control():
    """Constant-beta Fama-MacBeth recovers E[f] on zero-alpha assets, and a
    one-month misalignment breaks the recovery."""
    rng = np.random.default_rng(5)
    T, N, K = 400, 60, 4
    F = rng.normal(0.004, 0.04, (T, K))
    B = rng.normal(1.0, 0.5, (N, K))
    R = F @ B.T + rng.normal(0, 0.02, (T, N))
    lam, _ = sel.fmb(R, F)
    err = np.abs(lam[1:] - F.mean(axis=0)).max()
    assert err < 0.0005 and abs(lam[0]) < 0.0005
    Rs = np.roll(R, 1, axis=0)[1:]
    lam2, _ = sel.fmb(Rs, F[1:])
    assert np.abs(lam2[1:] - F[1:].mean(axis=0)).max() > 5 * err


# --------------------------------------------------------- on the JKP data ---
def test_leg_panel():
    requires_data()
    p = baseline.leg_panel('DEU')
    assert p['gvkey'].nunique() == 3 * len(baseline.CHARS)
    assert p['ret_usd_excess'].notna().all()
    assert p['datadate'].min() >= pd.Timestamp(baseline.START)
    assert p['ret_usd_excess'].abs().max() < 5
    ew = baseline.leg_panel('DEU', weighting='ew')
    assert not np.allclose(ew['ret_usd_excess'].to_numpy()[:500],
                           p['ret_usd_excess'].to_numpy()[:500])
    assert _raises(lambda: baseline.run_baseline('DEU', chars=['be_me'], verbose=False))


def test_gamma0_identity():
    """OLS passes through the means, so gamma0 = rbar - lambda' betabar on the
    estimation sample."""
    requires_data()
    r = baseline.run_baseline('DEU', window=60, verbose=False)
    lam, fc = r['lambdas'], r['factor_cols']
    months = lam.index.to_period('M')
    m = r['panel'].merge(r['betas'], on=['gvkey', 'iid', 'datadate'])
    m = m[m['datadate'].dt.to_period('M').isin(months)]
    implied = []
    for month, g in m.groupby(m['datadate'].dt.to_period('M')):
        lt = lam.loc[month.to_timestamp() + pd.offsets.MonthEnd(0)]
        b = g[['beta_' + c for c in fc]].mean().to_numpy()
        implied.append(g['ret_usd_excess'].mean() - lt[['lambda_' + c for c in fc]].to_numpy() @ b)
    assert abs(np.mean(implied) - lam['gamma0'].mean()) < 1e-10


def test_reliability_ordering():
    """Betas from longer windows are more reliable."""
    requires_data()
    out = {}
    cf = data.classic_factors('deu')
    panel = data.merge_monthly(baseline.leg_panel('DEU'), cf)
    for w in (36, 120):
        bs = fmb.rolling_betas_split(panel, list(cf.columns), window=w,
                                     min_periods=int(round(w * 2 / 3)))
        out[w] = fmb.split_half_correlation(bs, list(cf.columns)).mean()
    assert out[120] > out[36] > 0.10, out
