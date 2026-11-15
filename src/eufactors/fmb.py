"""Two-pass Fama-MacBeth estimation with rolling betas (Section 4.1).

Pass 1 estimates each test asset's betas on the `window` calendar months
strictly before the month they explain, so the beta used for month t never
contains the return of month t. Pass 2 regresses the month-t cross-section of
excess returns on these betas. The premium is the time-series mean of the
monthly slopes (equation 5).

Standard errors (Section 4.1.2 and Appendix B): Fama-MacBeth, the
multiplicative Shanken (1992) correction sqrt(1 + c) of equation (B2), and
Newey-West (1987) with Bartlett weights (equation B3). The reported test of
lambda = E[f] uses the monthly gaps lambda_t - f_t (equation B4).

The panel format is long: one row per test asset and month with the columns
gvkey (asset id), iid, datadate and ret_usd_excess, plus the factor columns.
"""
from math import exp, lgamma, log

import numpy as np
import pandas as pd


def _calendar_month(df, date_col='datadate'):
    """(year, month) key for grouping a cross-section by calendar month."""
    return df[date_col].dt.to_period('M')


def _lstsq(y, X1):
    """Least squares that returns None for a rank-deficient design instead of
    a minimum-norm solution."""
    coef, _res, rank, _sv = np.linalg.lstsq(X1, y, rcond=None)
    if rank < X1.shape[1]:
        return None
    return coef


def _ols_beta(y, X):
    """Slopes of an OLS regression with intercept, or None if rank-deficient."""
    X1 = np.column_stack([np.ones(len(X)), X])
    coef = _lstsq(y, X1)
    return None if coef is None else coef[1:]


def _ols_full(y, X):
    """[intercept, slopes] of an OLS regression, or None if rank-deficient."""
    X1 = np.column_stack([np.ones(len(X)), X])
    return _lstsq(y, X1)


# ------------------------------------------------------------------ pass 1 ---
def rolling_betas(panel, factor_cols, window=36, min_periods=24):
    """Rolling time-series betas, one row per asset and month.

    The window for month t covers the calendar months [t - window, t - 1], and
    an estimate requires at least `min_periods` observations inside it
    (Section 4.1.1: 40 of 60 months). `datadate` in the result is the month the
    beta is used to explain.
    """
    out = []
    cols = ['datadate', 'ret_usd_excess'] + factor_cols
    for (gvkey, iid), g in panel.sort_values('datadate').groupby(['gvkey', 'iid']):
        g = g[cols].dropna()
        n = len(g)
        if n < min_periods + 1:
            continue
        dates = g['datadate'].to_numpy()
        mnum = (g['datadate'].dt.year * 12 + g['datadate'].dt.month).to_numpy()
        y_all = g['ret_usd_excess'].to_numpy()
        X_all = g[factor_cols].to_numpy()
        for i in range(min_periods, n):
            lo = np.searchsorted(mnum, mnum[i] - window, side='left')
            y_win, X_win = y_all[lo:i], X_all[lo:i]
            if len(y_win) < min_periods:
                continue
            beta = _ols_beta(y_win, X_win)
            if beta is None:
                continue
            out.append((gvkey, iid, dates[i], len(y_win), *beta))
    cols_out = ['gvkey', 'iid', 'datadate', 'n_obs'] + [f'beta_{c}' for c in factor_cols]
    return pd.DataFrame(out, columns=cols_out)


def rolling_betas_split(panel, factor_cols, window=36, min_periods=24):
    """Two beta estimates per asset and month from the odd and even months of
    the same rolling window. Their cross-sectional correlation measures the
    reliability of the betas (Appendix C, cap of eight factors)."""
    out = []
    cols = ['datadate', 'ret_usd_excess'] + factor_cols
    min_half = len(factor_cols) + 3
    for (gvkey, iid), g in panel.sort_values('datadate').groupby(['gvkey', 'iid']):
        g = g[cols].dropna()
        n = len(g)
        if n < min_periods + 1:
            continue
        dates = g['datadate'].to_numpy()
        mnum = (g['datadate'].dt.year * 12 + g['datadate'].dt.month).to_numpy()
        y_all = g['ret_usd_excess'].to_numpy()
        X_all = g[factor_cols].to_numpy()
        for i in range(min_periods, n):
            lo = np.searchsorted(mnum, mnum[i] - window, side='left')
            idx = np.arange(lo, i)
            if len(idx) < min_periods:
                continue
            a_idx, b_idx = idx[0::2], idx[1::2]
            if len(a_idx) < min_half or len(b_idx) < min_half:
                continue
            beta_a = _ols_beta(y_all[a_idx], X_all[a_idx])
            beta_b = _ols_beta(y_all[b_idx], X_all[b_idx])
            if beta_a is None or beta_b is None:
                continue
            out.append((gvkey, iid, dates[i], len(a_idx), len(b_idx), *beta_a, *beta_b))
    cols_out = (['gvkey', 'iid', 'datadate', 'n_obs_a', 'n_obs_b']
                + [f'beta_a_{c}' for c in factor_cols] + [f'beta_b_{c}' for c in factor_cols])
    return pd.DataFrame(out, columns=cols_out)


def split_half_correlation(betas_split, factor_cols):
    """Per factor, the mean over months of the cross-sectional correlation
    between the two split-half betas. Months with fewer than ten assets are
    skipped."""
    month = _calendar_month(betas_split)
    counts = betas_split.groupby(month)['gvkey'].transform('size')
    usable = betas_split[counts >= 10]
    month_u = _calendar_month(usable)
    rows = {}
    for c in factor_cols:
        per_month = usable.groupby(month_u, group_keys=False).apply(
            lambda g, c=c: g['beta_a_' + c].corr(g['beta_b_' + c]))
        rows[c] = float(per_month.mean(skipna=True))
    return pd.Series(rows, name='corr_beta_a_beta_b')


# ------------------------------------------------------------------ pass 2 ---
def cross_sectional_lambdas(betas, panel, factor_cols, min_stocks=None,
                            intercept=True, group=None):
    """Monthly cross-sectional regressions of returns on betas (equation 4).

    Returns one row per month with the number of assets, gamma0 and the
    premia lambda_<factor>, indexed by month-end date.

    intercept=False restricts gamma0 to zero (Section 4.1.1, Table B4).
    group: optional function asset id -> group label. Returns and betas are
    then demeaned within group and month, which is equivalent to one intercept
    per group and month (pooled market, Table D1); gamma0 is NaN.
    """
    beta_cols = [f'beta_{c}' for c in factor_cols]
    k = len(factor_cols)
    min_stocks = min_stocks or max(10 * (k + 1), k + 5)

    merged = betas.merge(panel[['gvkey', 'iid', 'datadate', 'ret_usd_excess']],
                         on=['gvkey', 'iid', 'datadate'], how='inner')
    rows = []
    for month, g in merged.groupby(_calendar_month(merged)):
        g = g.dropna(subset=['ret_usd_excess'] + beta_cols)
        if len(g) < min_stocks:
            continue
        y = g['ret_usd_excess'].to_numpy()
        X = g[beta_cols].to_numpy()
        if group is not None:
            gm = (g.assign(_g=g['gvkey'].map(group))
                  .groupby('_g')[['ret_usd_excess'] + beta_cols].transform('mean'))
            slopes = _lstsq(y - gm['ret_usd_excess'].to_numpy(),
                            X - gm[beta_cols].to_numpy())
            coef = None if slopes is None else np.concatenate([[np.nan], slopes])
        elif intercept:
            coef = _ols_full(y, X)
        else:
            slopes = _lstsq(y, X)
            coef = None if slopes is None else np.concatenate([[0.0], slopes])
        if coef is None:
            continue
        rows.append((month.to_timestamp() + pd.offsets.MonthEnd(0), len(g), *coef))
    cols_out = ['datadate', 'n_stocks', 'gamma0'] + [f'lambda_{c}' for c in factor_cols]
    return pd.DataFrame(rows, columns=cols_out).set_index('datadate')


# -------------------------------------------------------- standard errors ---
def newey_west_lags(T):
    """Rule of thumb floor(4 (T/100)^(2/9)) (Appendix B): 5 lags at T = 344."""
    return int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))


def newey_west_se(x, lags=None):
    """Newey-West standard error of the mean of a series (equation B3).

    NaNs are dropped. lags defaults to newey_west_lags(T).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    T = len(x)
    if T < 3:
        return float('nan')
    if lags is None:
        lags = newey_west_lags(T)
    e = x - x.mean()
    gamma0 = (e @ e) / T
    var = gamma0
    for L in range(1, min(lags, T - 1) + 1):
        w = 1.0 - L / (lags + 1.0)
        gammaL = (e[L:] @ e[:-L]) / T
        var += 2.0 * w * gammaL
    var = max(var, 0.0)
    return float(np.sqrt(var / T))


def newey_west_cov(X, lags=None):
    """Newey-West covariance matrix of the mean of a (T x K) series, with the
    same weights and lag rule as newey_west_se. Rows with a NaN are dropped."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    X = X[~np.isnan(X).any(axis=1)]
    T = X.shape[0]
    if lags is None:
        lags = newey_west_lags(T)
    E = X - X.mean(axis=0)
    S = E.T @ E / T
    for L in range(1, min(lags, T - 1) + 1):
        w = 1.0 - L / (lags + 1.0)
        G = E[L:].T @ E[:-L] / T
        S += w * (G + G.T)
    return S / T


def chi2_sf(x, k):
    """Upper tail probability of the chi-squared distribution with k degrees
    of freedom (series expansion of the lower incomplete gamma function)."""
    if x <= 0:
        return 1.0
    a, term, total = k / 2.0, 1.0, 1.0
    for n in range(1, 2000):
        term *= (x / 2.0) / (a + n)
        total += term
        if term < 1e-14:
            break
    lower = exp(-x / 2.0 + a * log(x / 2.0) - lgamma(a + 1)) * total
    return max(0.0, 1.0 - lower)


def shanken_correction(lambdas_df, factor_panel, factor_cols, nw_lags_overlap=None):
    """Fama-MacBeth, Newey-West and Shanken-scaled standard errors of the premia.

    The Shanken factor is sqrt(1 + c) with c = m' Sigma_f^-1 m (equation B2).
    It is evaluated twice: at the estimated premia (shanken_scale) and at the
    factor means (shanken_scale_ef). The reported tests use the second
    (Section 4.1.2). nw_lags_overlap adds Newey-West standard errors with that
    many lags (the window overlap of 59 months in Table B1).
    """
    lam_cols = [f'lambda_{c}' for c in factor_cols]
    T = len(lambdas_df)
    lam_bar = lambdas_df[lam_cols].mean().to_numpy()
    se_raw = lambdas_df[lam_cols].std(ddof=1).to_numpy() / np.sqrt(T)

    common_dates = lambdas_df.index.intersection(factor_panel.index)
    Sigma_f = factor_panel.loc[common_dates, factor_cols].cov(ddof=1).to_numpy()
    c_shanken = float(lam_bar @ np.linalg.solve(Sigma_f, lam_bar))
    scale = float(np.sqrt(1 + c_shanken))
    ef_bar = factor_panel.loc[common_dates, factor_cols].mean().to_numpy()
    c_shanken_ef = float(ef_bar @ np.linalg.solve(Sigma_f, ef_bar))
    scale_ef = float(np.sqrt(1 + c_shanken_ef))

    se_shanken = se_raw * scale
    se_nw = np.array([newey_west_se(lambdas_df[c].to_numpy()) for c in lam_cols])
    se_nw_shanken = se_nw * scale
    se_nw_shanken_ef = se_nw * scale_ef
    if nw_lags_overlap is not None:
        se_nw_w = np.array([newey_west_se(lambdas_df[c].to_numpy(), lags=nw_lags_overlap)
                            for c in lam_cols])
    else:
        se_nw_w = np.full(len(lam_cols), np.nan)
    se_nw_w_shanken = se_nw_w * scale

    return pd.DataFrame({
        'factor': factor_cols,
        'lambda_mean': lam_bar,
        'se_raw': se_raw, 't_raw': lam_bar / se_raw,
        'se_shanken': se_shanken, 't_shanken': lam_bar / se_shanken,
        'se_nw': se_nw, 't_nw': lam_bar / se_nw,
        'se_nw_shanken': se_nw_shanken, 't_nw_shanken': lam_bar / se_nw_shanken,
        'se_nw_shanken_ef': se_nw_shanken_ef, 't_nw_shanken_ef': lam_bar / se_nw_shanken_ef,
        'se_nw_w': se_nw_w, 't_nw_w': lam_bar / se_nw_w,
        'se_nw_w_shanken': se_nw_w_shanken,
        't_nw_w_shanken': lam_bar / se_nw_w_shanken,
    }).assign(shanken_scale=scale, shanken_scale_ef=scale_ef, T=T,
              nw_lags_overlap=nw_lags_overlap)


# ---------------------------------------------------- tests of lambda = E[f] ---
def lambda_vs_factor_mean(lambdas_df, factor_panel, factor_cols, nw_lags=None,
                          shanken_scale=None):
    """Test of lambda = E[f] factor by factor (Section 4.1.2, equation B4).

    The test uses the gap series d_t = lambda_t - f_t over the months in
    which lambda is estimated, with i.i.d. and Newey-West standard errors.
    With shanken_scale, the Newey-West standard error is also multiplied by
    that factor (columns *_nw_sh); this is the reported t-statistic.
    """
    lam_cols = [f'lambda_{c}' for c in factor_cols]
    common = lambdas_df.index.intersection(factor_panel.index)
    lam = lambdas_df.loc[common, lam_cols]
    f = factor_panel.loc[common, factor_cols]
    T = len(common)

    rows = []
    for c in factor_cols:
        d = lam[f'lambda_{c}'].to_numpy() - f[c].to_numpy()
        d = d[~np.isnan(d)]
        se_iid = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float('nan')
        se_nw = newey_west_se(d, lags=nw_lags)
        se_sh = se_nw * shanken_scale if shanken_scale else float('nan')
        fj = f[c].to_numpy()
        fj = fj[~np.isnan(fj)]
        rows.append({
            'se_diff_nw_sh': se_sh,
            't_diff_nw_sh': float(np.nanmean(d) / se_sh) if se_sh == se_sh else float('nan'),
            'factor': c,
            'lambda_mean': float(np.nanmean(lam[f'lambda_{c}'])),
            'E_f': float(fj.mean()),
            't_E_f': float(fj.mean() / (fj.std(ddof=1) / np.sqrt(len(fj)))),
            'diff': float(np.nanmean(d)),
            'se_diff': se_iid, 't_diff': float(np.nanmean(d) / se_iid) if se_iid else float('nan'),
            'se_diff_nw': se_nw,
            't_diff_nw': float(np.nanmean(d) / se_nw) if se_nw and se_nw == se_nw else float('nan'),
        })
    return pd.DataFrame(rows).assign(T=T)


def wald_lambda_equals_ef(lambdas_df, factor_panel, factor_cols, nw_lags=None,
                          shanken_scale=None):
    """Joint Wald test of lambda = E[f] for all factors (Section 4.1.2).

    'wald' / 'p_value' use the i.i.d. covariance of the gap vector.
    'wald_nw_sh' / 'p_value_nw_sh' use its Newey-West covariance scaled by
    shanken_scale**2; this is the reported test. Both are chi2(K) under H0.
    """
    lam_cols = [f'lambda_{c}' for c in factor_cols]
    common = lambdas_df.index.intersection(factor_panel.index)
    D = (lambdas_df.loc[common, lam_cols].to_numpy()
         - factor_panel.loc[common, factor_cols].to_numpy())
    D = D[~np.isnan(D).any(axis=1)]
    T, K = D.shape
    d_bar = D.mean(axis=0)
    S = np.atleast_2d(np.cov(D, rowvar=False, ddof=1))
    try:
        stat = float(T * d_bar @ np.linalg.solve(S, d_bar))
    except np.linalg.LinAlgError:
        stat = float(T * d_bar @ np.linalg.pinv(S) @ d_bar)
    V = newey_west_cov(D, lags=nw_lags) * (shanken_scale or 1.0) ** 2
    try:
        stat_nw = float(d_bar @ np.linalg.solve(V, d_bar))
    except np.linalg.LinAlgError:
        stat_nw = float(d_bar @ np.linalg.pinv(V) @ d_bar)
    return {'wald': stat, 'df': K, 'p_value': chi2_sf(stat, K), 'T': T,
            'wald_nw_sh': stat_nw, 'p_value_nw_sh': chi2_sf(stat_nw, K)}


def attenuation_slope(lambdas_df, factor_panel, factor_cols):
    """Slope of lambda_t on f_t per factor (Section 4.1.3, Table B7, Panel B).

    With betas measured without error the monthly premium of a traded factor
    moves one for one with the factor return; a slope below one is consistent
    with attenuation.
    """
    common = lambdas_df.index.intersection(factor_panel.index)
    out = {}
    for c in factor_cols:
        lt = lambdas_df.loc[common, f'lambda_{c}'].to_numpy()
        ft = factor_panel.loc[common, c].to_numpy()
        ok = ~(np.isnan(lt) | np.isnan(ft))
        lt, ft = lt[ok], ft[ok]
        out[c] = float(np.cov(lt, ft)[0, 1] / np.var(ft, ddof=1))
    return pd.Series(out, name='attenuation_slope')
