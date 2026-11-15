"""Chapter 4: cross-sectional fit and the random-factor hurdle (Section 4.1.3,
Table 2 Panel B, Table B3 Panel B), following Lewellen, Nagel and Shanken (2010).

  L1  cross-sectional R^2 of average returns on full-sample betas (OLS, GLS
      and restricted to lambda = E[f], gamma0 = 0), with 95 percent intervals
      from a moving-block bootstrap (24-month blocks, 400 draws).
  L2  R^2 of 400 random models of k zero-investment portfolios of the test
      assets, and the share of them below the six-factor model (k = 6 is
      reported; k = 3 is the setting of Lewellen et al.).

Both are also computed for the 15 legs of the five classic characteristics.
These rows are not reported; they are kept because they draw from the same
random-number stream as the reported rows.

Writes L1_cross_sectional_fit.csv and L2_random_factor_hurdle.csv.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np                              # noqa: E402
import pandas as pd                             # noqa: E402

from eufactors import baseline, config, data    # noqa: E402

CLASSIC = list(data.CLASSICS)
N_DRAWS = 400
N_BOOT = 400
BLOCK = 24
SEED = 20260910


def balanced_panel(country, chars=None, start='1994-01-01'):
    cf = data.classic_factors(country.lower())
    legs = baseline.leg_panel(country, chars=chars, start=start)
    R = legs.pivot_table(index='datadate', columns='gvkey', values='ret_usd_excess')
    R.index = pd.PeriodIndex(R.index, freq='M')
    cf.index = pd.PeriodIndex(cf.index, freq='M')
    idx = R.dropna(axis=1, how='all').dropna().index.intersection(cf.dropna().index)
    return R.loc[idx].dropna(axis=1), cf.loc[idx]


def cs_fit(R, F):
    """Cross-sectional R^2 (OLS, GLS, restricted) and pricing errors.

    Betas come from full-sample time-series regressions. GLS weights by the
    inverse return covariance, shrunk 10 percent toward its diagonal because
    N is large relative to T.
    """
    Rv = np.asarray(R, float)
    Fv = np.asarray(F, float)
    T, N = Rv.shape
    X = np.column_stack([np.ones(T), Fv])
    coef = np.linalg.lstsq(X, Rv, rcond=None)[0]
    B = coef[1:].T
    alpha_ts = coef[0]
    rbar = Rv.mean(axis=0)
    Z = np.column_stack([np.ones(N), B])
    fit = Z @ np.linalg.lstsq(Z, rbar, rcond=None)[0]
    e = rbar - fit
    sst = ((rbar - rbar.mean()) ** 2).sum()
    r2_ols = 1.0 - (e ** 2).sum() / sst if sst > 0 else np.nan
    # restricted: premia fixed at the factor means, no intercept
    e_r = rbar - B @ Fv.mean(axis=0)
    r2_restricted = 1.0 - (e_r ** 2).sum() / sst if sst > 0 else np.nan

    S = np.cov(Rv, rowvar=False)
    lam = 0.10
    S = (1 - lam) * S + lam * np.diag(np.diag(S))
    try:
        Si = np.linalg.inv(S)
        d = rbar - rbar.mean()
        Zg = Si @ Z
        fit_g = Z @ np.linalg.lstsq(Z.T @ Zg, Zg.T @ rbar, rcond=None)[0]
        eg = rbar - fit_g
        r2_gls = 1.0 - float(eg @ Si @ eg) / float(d @ Si @ d)
    except np.linalg.LinAlgError:
        r2_gls = np.nan
    return dict(r2_ols=r2_ols, r2_gls=r2_gls, r2_restricted=r2_restricted,
                sse_pricing_pct2=float((e ** 2).sum()) * 1e4,
                mean_abs_alpha_ts_pct=float(np.abs(alpha_ts).mean()) * 100)


def block_bootstrap_ci(R, F, stat='r2_ols', n=N_BOOT, block=BLOCK, rng=None):
    """2.5 and 97.5 percentiles from a moving-block bootstrap over months."""
    rng = rng or np.random.default_rng(SEED)
    Rv, Fv = np.asarray(R, float), np.asarray(F, float)
    T = Rv.shape[0]
    nb = int(np.ceil(T / block))
    vals = []
    for _ in range(n):
        starts = rng.integers(0, T - block + 1, size=nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:T]
        v = cs_fit(Rv[idx], Fv[idx])[stat]
        if np.isfinite(v):
            vals.append(v)
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def random_factor_r2(R, k=3, n=N_DRAWS, rng=None):
    """OLS R^2 of n random models, each of k zero-investment portfolios of the
    test assets with random weights that sum to zero."""
    rng = rng or np.random.default_rng(SEED)
    Rv = np.asarray(R, float)
    N = Rv.shape[1]
    out = []
    for _ in range(n):
        W = rng.normal(size=(N, k))
        W -= W.mean(axis=0)
        out.append(cs_fit(Rv, Rv @ W)['r2_ols'])
    return np.array([x for x in out if np.isfinite(x)])


def main():
    rng = np.random.default_rng(SEED)
    rows_fit, rows_rand = [], []
    for c in config.COUNTRIES:
        for label, chars in (('test assets (168 legs)', None),
                             ('classic legs (15)', CLASSIC)):
            R, cf = balanced_panel(c, chars=chars)
            F = cf.to_numpy()
            f = cs_fit(R, F)
            lo_o, hi_o = block_bootstrap_ci(R, F, 'r2_ols', rng=rng)
            lo_g, hi_g = block_bootstrap_ci(R, F, 'r2_gls', rng=rng)
            lo_r, hi_r = block_bootstrap_ci(R, F, 'r2_restricted', rng=rng)
            rows_fit.append(dict(country=c, assets=label, N=R.shape[1], T=len(R),
                                 r2_ols=f['r2_ols'], r2_ols_lo=lo_o, r2_ols_hi=hi_o,
                                 r2_gls=f['r2_gls'], r2_gls_lo=lo_g, r2_gls_hi=hi_g,
                                 r2_restricted=f['r2_restricted'],
                                 r2_restricted_lo=lo_r, r2_restricted_hi=hi_r,
                                 sse_pricing_pct2=f['sse_pricing_pct2'],
                                 mean_abs_alpha_pct=f['mean_abs_alpha_ts_pct']))
            for k in (6, 3):
                rr = random_factor_r2(R, k=k, rng=rng)
                rows_rand.append(dict(country=c, assets=label, k=k, N=R.shape[1],
                                      n_draws=len(rr),
                                      model_r2_ols=f['r2_ols'],
                                      random_median=float(np.median(rr)),
                                      random_p90=float(np.percentile(rr, 90)),
                                      random_p99=float(np.percentile(rr, 99)),
                                      draws_beaten=int((rr < f['r2_ols']).sum()),
                                      model_beats_random_pct=float((rr < f['r2_ols']).mean() * 100)))
                print('  %s %-24s k=%d R2_OLS %.3f [%.2f, %.2f] | random median %.3f p90 %.3f'
                      % (c, label, k, f['r2_ols'], lo_o, hi_o,
                         np.median(rr), np.percentile(rr, 90)), flush=True)

    pd.DataFrame(rows_fit).set_index(['country', 'assets']).to_csv(
        config.out('L1_cross_sectional_fit.csv'))
    pd.DataFrame(rows_rand).set_index(['country', 'assets', 'k']).to_csv(
        config.out('L2_random_factor_hurdle.csv'))
    print('-> %s' % config.OUTPUT)


if __name__ == '__main__':
    main()
