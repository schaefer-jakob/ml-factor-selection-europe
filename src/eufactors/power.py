"""Power and minimum detectable effects of the baseline tests (Section 4.1.2,
Table 2 power columns and Table B3, Panel A)."""
from math import erf, sqrt

import numpy as np
import pandas as pd

from . import fmb

Z_ALPHA = 1.959963985    # two-sided 5 percent
Z_BETA = 0.8416212336    # 80 percent power


def _power(ncp, z_a):
    """Power of a two-sided z-test with non-centrality ncp."""
    return (1 - 0.5 * (1 + erf((z_a - ncp) / sqrt(2)))
            + 0.5 * (1 + erf((-z_a - ncp) / sqrt(2))))


def _norm_ppf(p):
    """Standard normal quantile (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q, r = p - 0.5, (p - 0.5) ** 2
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def power_analysis(out, alpha=0.05, target=None, nw_lags=None, shanken_scale=None):
    """Minimum detectable effect at 80 percent power and power against the
    factor mean, per factor of a run_baseline() result.

    Columns without suffix refer to the test lambda = 0 with the Newey-West
    standard error of lambda_t. The *_diff columns refer to the reported test
    lambda = E[f], with the Newey-West standard error of lambda_t - f_t; their
    power is against the alternative that the cross-section pays no premium,
    a gap of |E[f]|. shanken_scale multiplies both standard errors (*_sh).
    """
    lam = out['lambdas']
    fcols = out['factor_cols']
    cf = out['factor_panel']
    z_a = Z_ALPHA if abs(alpha - 0.05) < 1e-9 else _norm_ppf(1 - alpha / 2)
    z_b = Z_BETA

    rows = []
    for c in fcols:
        series = lam[f'lambda_{c}'].dropna().to_numpy()
        T = len(series)
        se_iid = series.std(ddof=1) / np.sqrt(T)
        se = fmb.newey_west_se(series, lags=nw_lags)
        lags_used = fmb.newey_west_lags(T) if nw_lags is None else int(nw_lags)
        mde = (z_a + z_b) * se
        cf_a = cf.loc[cf.index.intersection(lam.index), c]
        tgt = float(cf_a.mean()) if target is None else float(target)
        ncp = abs(tgt) / se if se > 0 else float('inf')
        power = _power(ncp, z_a)
        d = (lam.loc[cf_a.index, f'lambda_{c}'] - cf_a).dropna().to_numpy()
        se_d = fmb.newey_west_se(d, lags=nw_lags)
        sc = shanken_scale or 1.0
        rows.append({'factor': c, 'T': T, 'se_lambda': se,
                     'mde_80pct': mde, 'target_premium': tgt,
                     'power_vs_target': power,
                     'lambda_hat': float(series.mean()),
                     'nw_lags': lags_used,
                     'se_nw_over_iid': (se / se_iid) if se_iid > 0 else float('nan'),
                     'se_lambda_sh': se * sc,
                     'mde_80pct_sh': (z_a + z_b) * se * sc,
                     'power_vs_target_sh': _power(abs(tgt) / (se * sc), z_a),
                     'se_diff': se_d,
                     'se_diff_sh': se_d * sc,
                     'mde_80pct_diff_sh': (z_a + z_b) * se_d * sc,
                     'power_diff_sh': _power(abs(tgt) / (se_d * sc), z_a)})
    return pd.DataFrame(rows)
