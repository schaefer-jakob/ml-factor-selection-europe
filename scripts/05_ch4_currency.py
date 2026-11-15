"""Chapter 4 and Appendix D: currency checks of the baseline.

  FX0  conversion inputs per country (exchange-rate and interest-rate moments)
  FX1  single markets: US dollar as published and local currency with local
       interest rates (Table B5, local-currency row of Table 3)
  FX2  pooled market: one common intercept and one intercept per country and
       month (Table D1)

Window 60, Newey-West standard errors times the Shanken factor at E[f].
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402

from eufactors import baseline, config, data, fx    # noqa: E402

WINDOW = 60


def summarise(res, label, krs_defined=True):
    """One row per factor and one model row. krs_defined=False leaves the KRS
    column empty: krs.py estimates one common intercept, so it does not apply
    to the run with country-month intercepts."""
    L = res['lam_vs_ef'].set_index('factor')
    k = res['krs']
    fac = res['factor_cols']
    rows = []
    for i, f in enumerate(fac):
        rows.append(dict(spec=label, item=f,
                         lam_pct=100 * L.loc[f, 'lambda_mean'],
                         ef_pct=100 * L.loc[f, 'E_f'],
                         t_lam_eq_ef=L.loc[f, 't_diff_nw_sh'],
                         t_krs_lam_eq_ef=(k['diff'][i] / k['se_diff'][i]
                                          if krs_defined else np.nan)))
    g0 = res['gamma0']
    w = res['wald_ef']
    g = res['grs']
    rows.append(dict(spec=label, item='model',
                     gamma0_pct=100 * g0['mean'], gamma0_t=g0['t_nw_sh'],
                     wald=w['wald_nw_sh'], wald_p=w['p_value_nw_sh'],
                     grs=g['grs'], grs_p=g['p_value'],
                     mean_abs_alpha_pct=g['mean_abs_alpha_pct_mo'],
                     beta_mkt_full_mean=g.get('beta_mkt_full_mean', np.nan),
                     beta_mkt_full_sd=g.get('beta_mkt_full_sd', np.nan),
                     T_fmb=g0['T'], T_grs=g['T']))
    return rows


def main():
    countries = list(config.COUNTRIES)
    single = [c for c in countries if c.lower() not in data.POOLS]
    pools = [c for c in countries if c.lower() in data.POOLS]

    inputs = []
    for c in sorted({m for p in pools for m in data.POOLS[p.lower()]} | {c.lower() for c in single}):
        t = fx.conversion_table(c)
        t = t[(t.index >= pd.Period('1994-01', 'M')) & (t.index <= pd.Period('2025-12', 'M'))]
        inputs.append(dict(country=c.upper(), first=str(t.index.min()), last=str(t.index.max()),
                           T=len(t), s_mean_pct=100 * t.s.mean(), s_sd_pct=100 * t.s.std(),
                           x_mean_pct=100 * t.x.mean(), x_sd_pct=100 * t.x.std(),
                           rf_loc_mean_pct=100 * t.rf_loc.mean(),
                           rf_us_mean_pct=100 * t.rf_us.mean()))
    pd.DataFrame(inputs).set_index('country').to_csv(config.out('FX0_conversion_inputs.csv'))

    rows = []
    for c in single:
        for cur in ('usd', 'local'):
            res = baseline.run_baseline(c, window=WINDOW, verbose=True, currency=cur)
            for r in summarise(res, cur):
                r['country'] = c
                rows.append(r)
    if rows:
        T1 = pd.DataFrame(rows).set_index(['country', 'spec', 'item'])
        T1.to_csv(config.out('FX1_local_currency.csv'))
        print(T1.round(3).to_string())

    rows = []
    for c in pools:
        for label, dm in (('usd_common_intercept', False), ('usd_member_intercepts', True)):
            res = baseline.run_baseline(c, window=WINDOW, verbose=True, demean_by_member=dm)
            for r in summarise(res, label, krs_defined=not dm):
                r['country'] = c
                rows.append(r)
    if rows:
        T2 = pd.DataFrame(rows).set_index(['country', 'spec', 'item'])
        T2.to_csv(config.out('FX2_pooled_member_intercepts.csv'))
        print(T2.round(3).to_string())


if __name__ == '__main__':
    main()
