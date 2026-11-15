"""The six-factor baseline on JKP tercile portfolios (Chapter 4).

Test assets are the three tercile portfolios (legs) of the 56 characteristics
in config/selected_characteristics.csv, 168 per market (Section 3.2). Factors
are the JKP country versions of MKT, SMB, HML, RMW, CMA and WML, or the French
Europe factors in the robustness grid.
"""
import zipfile
from math import exp, lgamma, log

import numpy as np
import pandas as pd

from . import config, data, fmb, krs

CHARS = pd.read_csv(config.config('selected_characteristics.csv'))['characteristic'].tolist()
START = '1994-01-01'
WINDOWS = [60, 36, 120]          # 60 months is the baseline (Section 4.1.1)


def min_assets(n_assets, k):
    """Smallest cross-section kept in pass 2: 75 percent of the legs, at
    least k + 5, never more than the panel itself."""
    return min(max(k + 5, int(round(0.75 * n_assets))), n_assets)


def _read_zip(path):
    with zipfile.ZipFile(path) as z:
        return pd.read_csv(z.open(z.namelist()[0]))


def leg_path(country, char, weighting='vw_cap'):
    return config.data_path('jkp_portfolios', cty=country.lower(), char=char, weight=weighting)


def leg_panel(country, chars=None, start=START, weighting='vw_cap'):
    """Test-asset returns in the long panel format of fmb.

    The asset id is '<characteristic>_pf<n>' (n = 1, 2, 3). A pooled market
    (data.POOLS) stacks the legs of its members and prefixes the id with the
    member code, so the Swiss and Swedish legs stay separate assets.
    """
    chars = chars or CHARS
    members = data.POOLS.get(country.lower())
    frames = []
    for c in (members or (country.lower(),)):
        for ch in chars:
            df = _read_zip(leg_path(c, ch, weighting))
            df['datadate'] = pd.to_datetime(df['date'])
            df['gvkey'] = ((c + '.' if members else '') + df['name']
                           + '_pf' + df['pf'].astype(int).astype(str))
            frames.append(df[['gvkey', 'datadate', 'ret']]
                          .rename(columns={'ret': 'ret_usd_excess'}))
    legs = pd.concat(frames, ignore_index=True)
    legs['iid'] = '01'
    legs = legs.dropna(subset=['ret_usd_excess'])
    legs = legs[legs['datadate'] >= start]
    return legs.sort_values(['gvkey', 'datadate']).reset_index(drop=True)


# ---------------------------------------------------------------- GRS test ---
def _betacf(a, b, x, itmax=300, eps=3e-16):
    """Continued fraction of the incomplete beta function (Lentz)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x)
    front = exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def f_sf(f, d1, d2):
    """P(F > f) for the F distribution with d1 and d2 degrees of freedom."""
    if f <= 0:
        return 1.0
    return _betainc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))


def grs_test(R, F):
    """Gibbons, Ross and Shanken (1989) test that all N time-series intercepts
    are zero (Section 4.1.2).

        GRS = (T/N) (T-N-K)/(T-K-1) a' S^-1 a / (1 + mu' Omega^-1 mu)
            ~ F(N, T-N-K)

    with the maximum-likelihood factor covariance Omega. R is (T x N), F is
    (T x K). Also returns the mean absolute intercept and the full-sample betas.
    """
    R = np.asarray(R, float)
    F = np.asarray(F, float)
    T, N = R.shape
    K = F.shape[1]
    if N >= T - K:
        raise ValueError('GRS needs N < T - K; got N=%d, T=%d, K=%d' % (N, T, K))
    X = np.column_stack([np.ones(T), F])
    coef, *_ = np.linalg.lstsq(X, R, rcond=None)
    alpha = coef[0]
    resid = R - X @ coef
    Sigma = resid.T @ resid / (T - K - 1)
    mu = F.mean(axis=0)
    Fc = F - mu
    Omega = Fc.T @ Fc / T
    a_term = float(alpha @ np.linalg.solve(Sigma, alpha))
    sr2 = float(mu @ np.linalg.solve(Omega, mu))
    stat = (T / N) * ((T - N - K) / (T - K - 1)) * a_term / (1.0 + sr2)
    # mean |alpha| relative to the mean absolute deviation of average returns
    rbar = R.mean(axis=0)
    a_abs_r = float(np.abs(rbar - rbar.mean()).mean())
    return {
        'grs': stat,
        'df1': N,
        'df2': T - N - K,
        'p_value': f_sf(stat, N, T - N - K),
        'mean_abs_alpha_pct_mo': float(np.abs(alpha).mean() * 100),
        'aa_over_ar': float(np.abs(alpha).mean() / a_abs_r) if a_abs_r > 0 else float('nan'),
        'betas_full': coef[1:],
        'sharpe2_factors': sr2,
        'T': T, 'N': N, 'K': K,
    }


def grs_or_nan(R, F):
    """grs_test where it is defined. Where N >= T - K (the pooled market with
    336 legs, Appendix D) the test fields are NaN and the descriptive fields
    are still computed."""
    try:
        return grs_test(R, F)
    except ValueError as e:
        if 'GRS needs N < T - K' not in str(e):
            raise
        R = np.asarray(R, float)
        F = np.asarray(F, float)
        T, N = R.shape
        K = F.shape[1]
        X = np.column_stack([np.ones(T), F])
        coef = np.linalg.lstsq(X, R, rcond=None)[0]
        alpha = coef[0]
        rbar = R.mean(axis=0)
        a_abs_r = float(np.abs(rbar - rbar.mean()).mean())
        return {'grs': float('nan'), 'df1': N, 'df2': T - N - K,
                'p_value': float('nan'),
                'mean_abs_alpha_pct_mo': float(np.abs(alpha).mean() * 100),
                'aa_over_ar': float(np.abs(alpha).mean() / a_abs_r) if a_abs_r > 0 else float('nan'),
                'betas_full': coef[1:], 'sharpe2_factors': float('nan'),
                'T': T, 'N': N, 'K': K, 'not_defined': str(e)}


# ----------------------------------------------------------------- pipeline ---
def run_baseline(country, window=60, min_periods=None, chars=None, start=START,
                 verbose=True, weighting='vw_cap', factor_source='jkp',
                 currency='usd', demean_by_member=False):
    """Fama-MacBeth estimation of the six-factor model for one market.

    window: beta window in months (60 baseline; 36 and 120 in the grid).
    weighting: 'vw_cap' (baseline) or 'ew' test assets.
    factor_source: 'jkp' (baseline) or 'french'.
    currency: 'usd' (baseline) or 'local' (Table B5, see fx.py).
    demean_by_member: pooled market only, one intercept per country and month
        (Table D1).
    """
    min_periods = min_periods or int(round(window * 2.0 / 3.0))
    cf = (data.classic_factors(country.lower()) if factor_source == 'jkp'
          else data.french_classic_factors())
    factor_cols = list(cf.columns)

    legs = leg_panel(country, chars=chars, start=start, weighting=weighting)
    if currency == 'local':
        if country.lower() in data.POOLS:
            raise ValueError('currency="local" is defined per country, not for a '
                             'pool; use demean_by_member for %s' % country)
        from . import fx
        legs = fx.legs_to_local(legs, country)
        cf = fx.factors_to_local(cf, country)
    elif currency != 'usd':
        raise ValueError('currency must be "usd" or "local"')
    panel = data.merge_monthly(legs, cf)
    floor = min_assets(panel['gvkey'].nunique(), len(factor_cols))

    if panel['gvkey'].nunique() < len(factor_cols) + 3:
        raise ValueError(
            'need at least k+3 = %d test assets for a %d-factor cross-section, '
            'got %d' % (len(factor_cols) + 3, len(factor_cols),
                        panel['gvkey'].nunique()))

    betas = fmb.rolling_betas(panel, factor_cols, window=window,
                              min_periods=min_periods)
    group = None
    if demean_by_member:
        if country.lower() not in data.POOLS:
            raise ValueError('demean_by_member needs a pooled pseudo-country')
        group = lambda g: g.split('.', 1)[0]     # noqa: E731  'che.<name>_pf1' -> 'che'
    lambdas = fmb.cross_sectional_lambdas(betas, panel, factor_cols,
                                          min_stocks=floor, group=group)
    if len(lambdas) == 0:
        raise ValueError(
            'no month reached the floor of %d assets (panel has %d legs)'
            % (floor, panel['gvkey'].nunique()))

    # Standard errors of Section 4.1.2: Newey-West with the automatic lag,
    # times the Shanken factor evaluated at E[f]. The 59-lag column feeds
    # Table B1 only.
    shanken = fmb.shanken_correction(lambdas, cf, factor_cols,
                                     nw_lags_overlap=window - 1)
    scale_ef = float(shanken['shanken_scale_ef'].iloc[0])
    lam_vs_ef = fmb.lambda_vs_factor_mean(lambdas, cf, factor_cols,
                                          shanken_scale=scale_ef)
    wald_ef = fmb.wald_lambda_equals_ef(lambdas, cf, factor_cols,
                                        shanken_scale=scale_ef)
    slope = fmb.attenuation_slope(lambdas, cf, factor_cols)

    g0 = lambdas['gamma0']
    gamma0 = {
        'mean': float(g0.mean()),
        'se_iid': float(g0.std(ddof=1) / np.sqrt(len(g0))),
        'se_nw': fmb.newey_west_se(g0.to_numpy()),
        'se_nw_w': fmb.newey_west_se(g0.to_numpy(), lags=window - 1),
        'T': int(len(g0)),
    }
    gamma0['se_nw_sh'] = gamma0['se_nw'] * scale_ef
    for k in ('iid', 'nw', 'nw_w', 'nw_sh'):
        gamma0['t_' + k] = gamma0['mean'] / gamma0['se_' + k]

    # gamma0 restricted to zero (Section 4.1.1, Table B4)
    lambdas_noint = fmb.cross_sectional_lambdas(betas, panel, factor_cols,
                                                min_stocks=floor, intercept=False)
    lam_vs_ef_noint = fmb.lambda_vs_factor_mean(lambdas_noint, cf, factor_cols,
                                                shanken_scale=scale_ef)
    mkt_col = 'beta_MKT' if 'beta_MKT' in betas.columns else None
    beta_mkt_cs_sd = (float(betas.groupby('datadate')[mkt_col].std().mean())
                      if mkt_col else float('nan'))

    # GRS on the balanced panel of months in which every leg has a return
    wide = panel.pivot_table(index='datadate', columns='gvkey',
                             values='ret_usd_excess')
    fac = cf.copy()
    fac.index = pd.PeriodIndex(fac.index, freq='M')
    wide.index = pd.PeriodIndex(wide.index, freq='M')
    joined = wide.join(fac, how='inner')
    both = joined.dropna()
    grs = grs_or_nan(both[wide.columns].to_numpy(), both[factor_cols].to_numpy())
    grs['months_dropped'] = int(len(joined) - len(both))
    grs['first_month'] = str(both.index.min())
    grs['last_month'] = str(both.index.max())
    if 'MKT' in factor_cols:
        grs['beta_mkt_full_sd'] = float(
            grs['betas_full'][factor_cols.index('MKT')].std(ddof=1))
        grs['beta_mkt_full_mean'] = float(
            grs['betas_full'][factor_cols.index('MKT')].mean())

    # Kan, Robotti and Shanken (2013) on full-sample betas, same balanced panel
    krs_out = krs.krs_standard_errors(both[wide.columns].to_numpy(),
                                      both[factor_cols].to_numpy())

    # factor means over exactly the months in which lambda is estimated
    ef_aligned = cf.loc[cf.index.intersection(lambdas.index), factor_cols].mean()

    out = dict(country=country, window=window, min_periods=min_periods,
               n_assets=panel['gvkey'].nunique(), factor_cols=factor_cols,
               E_f=ef_aligned,
               panel=panel, betas=betas, lambdas=lambdas, shanken=shanken,
               lam_vs_ef=lam_vs_ef, wald_ef=wald_ef, gamma0=gamma0, grs=grs,
               shanken_scale_ef=scale_ef, attenuation_slope=slope,
               lambdas_noint=lambdas_noint, lam_vs_ef_noint=lam_vs_ef_noint,
               beta_mkt_cs_sd=beta_mkt_cs_sd, krs=krs_out,
               factors=cf, factor_panel=cf,
               spec=dict(country=country, window=window, weighting=weighting,
                         factor_source=factor_source, start=start,
                         currency=currency, demean_by_member=demean_by_member,
                         n_chars=len(chars or CHARS)))

    if verbose:
        print('  %s w=%-3d  N=%d legs, T=%d months, GRS=%.2f (p=%.4f)'
              % (country, window, out['n_assets'], gamma0['T'],
                 grs['grs'], grs['p_value']), flush=True)
    return out
