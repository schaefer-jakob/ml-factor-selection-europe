"""Chapter 4: Fama-MacBeth estimates of the six-factor model.

Writes to output/<scope>/:
  T1_sample.csv, T1_leg_moments_<CTY>.csv   sample and test-asset returns (Section 3.2, Table B7 Panel A)
  T3_premia_<CTY>.csv                       premia, E[f], t(lambda = E[f]), attenuation slope (Tables 2, B7)
  T4_model_tests.csv                        Wald, GRS, gamma0, mean |alpha| (Table 2, Panel B)
  T5_power.csv                              power and minimum detectable effects (Tables 2, B3)
  T7_robustness_grid.csv                    nine design variations per market (Tables 3, B2)
  T11_gamma0_zero.csv                       gamma0 restricted to zero (Tables 3, B4)
  T12_krs.csv                               full-sample betas, KRS standard errors (Tables 3, B4)
  B1_standard_errors.csv                    components of the standard errors (Table B1)
  premia_ratio_by_window.csv                lambda / E[f] by beta window (Table 3)
"""
import gc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import pandas as pd                                  # noqa: E402

from eufactors import baseline, config, fmb, power   # noqa: E402

HEADLINE = 60


def save(name, df):
    df.to_csv(config.out(name + '.csv'))
    return df


def sample_description(res):
    p = res['panel']
    month = p['datadate'].dt.to_period('M')
    per_leg = p.groupby('gvkey')['ret_usd_excess'].agg(['mean', 'std', 'count'])
    sample = pd.Series({
        'country': res['spec']['country'],
        'weighting': res['spec']['weighting'],
        'first_month': str(month.min()), 'last_month': str(month.max()),
        'n_months': int(month.nunique()),
        'n_test_assets': int(p['gvkey'].nunique()),
        'n_characteristics': res['spec']['n_chars'],
        'asset_months': int(len(p)),
        'excess_ret_mean_pct': 100 * float(p['ret_usd_excess'].mean()),
        'excess_ret_sd_pct': 100 * float(p['ret_usd_excess'].std()),
        'leg_mean_min_pct': 100 * float(per_leg['mean'].min()),
        'leg_mean_max_pct': 100 * float(per_leg['mean'].max()),
        'leg_mean_spread_pct': 100 * float(per_leg['mean'].max() - per_leg['mean'].min()),
    })
    return sample, per_leg


def standard_error_components(res, overlap_lags):
    """Table B1: how the i.i.d., Newey-West and Shanken components change the
    test of lambda = E[f]."""
    lam, cf, fc = res['lambdas'], res['factors'], res['factor_cols']
    sh = res['shanken'].set_index('factor')
    ed = res['lam_vs_ef'].set_index('factor')
    scale_l = float(sh['shanken_scale'].iloc[0])
    scale_e = float(sh['shanken_scale_ef'].iloc[0])
    common = lam.index.intersection(cf.index)
    rows = []
    for f in fc:
        lt = lam.loc[common, 'lambda_' + f]
        ft = cf.loc[common, f]
        d = (lt - ft).dropna().to_numpy()
        se_nw_long = fmb.newey_west_se(d, lags=overlap_lags)
        rows.append(dict(
            factor=f,
            var_lambda_over_var_f=float(lt.var(ddof=1) / ft.var(ddof=1)),
            nw_over_iid=float(sh.loc[f, 'se_nw'] / sh.loc[f, 'se_raw']),
            nw_long_over_iid=float(sh.loc[f, 'se_nw_w'] / sh.loc[f, 'se_raw']),
            t_iid=float(ed.loc[f, 't_diff']),
            t_nw=float(ed.loc[f, 't_diff_nw']),
            t_nw_sh_lambda=float(ed.loc[f, 't_diff_nw'] / scale_l),
            t_nw_sh_ef=float(ed.loc[f, 't_diff_nw_sh']),
            t_nw_long_sh_ef=float(d.mean() / (se_nw_long * scale_e)),
            shanken_scale_lambda=scale_l, shanken_scale_ef=scale_e,
            nw_lags=fmb.newey_west_lags(len(d)), nw_long_lags=overlap_lags))
    return pd.DataFrame(rows)


def main():
    countries = config.COUNTRIES
    head = {c: baseline.run_baseline(c, window=HEADLINE, verbose=True) for c in countries}

    # --- sample and test-asset returns -------------------------------------
    samples = {}
    for c, r in head.items():
        s, per_leg = sample_description(r)
        samples[c] = s
        save('T1_leg_moments_%s' % c, per_leg)
    save('T1_sample', pd.DataFrame(samples))

    # --- premia and model tests (Table 2) -----------------------------------
    tests = []
    for c, r in head.items():
        sh = r['shanken'].set_index('factor')
        ef = r['lam_vs_ef'].set_index('factor')
        t = pd.DataFrame({
            'lambda_pct_mo': 100 * sh['lambda_mean'],
            't_shanken': sh['t_shanken'],
            't_nw_shanken': sh['t_nw_shanken'],
            't_nw_sh_ef': sh['t_nw_shanken_ef'],
            'E_f_pct_mo': 100 * r['E_f'],
        })
        t['diff_pct_mo'] = t['lambda_pct_mo'] - t['E_f_pct_mo']
        t['t_diff_nw'] = ef['t_diff_nw']
        t['t_diff_nw_sh'] = ef['t_diff_nw_sh']
        t['t_E_f'] = ef['t_E_f']
        t['attenuation_slope'] = r['attenuation_slope']
        t['shanken_scale_ef'] = r['shanken_scale_ef']
        save('T3_premia_%s' % c, t)
        w = r['wald_ef']
        tests.append(dict(country=c, grs=r['grs']['grs'], grs_df1=r['grs']['df1'],
                          grs_df2=r['grs']['df2'], grs_p=r['grs']['p_value'],
                          mean_abs_alpha_pct=r['grs']['mean_abs_alpha_pct_mo'],
                          aa_over_ar=r['grs']['aa_over_ar'],
                          wald=w.get('wald'), wald_df=w.get('df'),
                          wald_p=w.get('p_value'),
                          wald_nw_sh=w.get('wald_nw_sh'),
                          wald_p_nw_sh=w.get('p_value_nw_sh'),
                          gamma0_pct=100 * r['gamma0']['mean'],
                          gamma0_t_nw=r['gamma0']['t_nw'],
                          gamma0_t_nw_sh=r['gamma0']['t_nw_sh'],
                          beta_mkt_rolling_cs_sd=r['beta_mkt_cs_sd'],
                          beta_mkt_full_sd=r['grs'].get('beta_mkt_full_sd'),
                          beta_mkt_full_mean=r['grs'].get('beta_mkt_full_mean'),
                          T=r['gamma0']['T']))
    save('T4_model_tests', pd.DataFrame(tests).set_index('country'))

    # --- gamma0 restricted to zero (Table B4) -------------------------------
    rows = []
    for c, r in head.items():
        e = r['lam_vs_ef_noint'].set_index('factor')
        for f in r['factor_cols']:
            rows.append(dict(country=c, factor=f,
                             lambda_pct_mo=100 * e.loc[f, 'lambda_mean'],
                             E_f_pct_mo=100 * e.loc[f, 'E_f'],
                             t_diff_nw_sh=e.loc[f, 't_diff_nw_sh']))
    save('T11_gamma0_zero', pd.DataFrame(rows).set_index(['country', 'factor']))

    # --- full-sample betas, KRS standard errors (Table B4) ------------------
    rows = []
    for c, r in head.items():
        k = r['krs']
        names = ['gamma0'] + r['factor_cols']
        for i, f in enumerate(names):
            row = dict(country=c, factor=f,
                       estimate_pct_mo=100 * k['gamma'][i],
                       t_krs=k['gamma'][i] / k['se_gamma'][i])
            if i > 0:
                row['t_diff_krs'] = k['diff'][i - 1] / k['se_diff'][i - 1]
            rows.append(row)
    save('T12_krs', pd.DataFrame(rows).set_index(['country', 'factor']))

    # --- power (Tables 2 and B3) -------------------------------------------
    pw_rows = []
    for c, r in head.items():
        d = power.power_analysis(r, shanken_scale=r['shanken_scale_ef'])
        d.insert(0, 'country', c)
        pw_rows.append(d)
    P = pd.concat(pw_rows, ignore_index=True)
    pct = ('se_lambda', 'mde_80pct', 'target_premium', 'lambda_hat', 'se_lambda_sh',
           'mde_80pct_sh', 'se_diff', 'se_diff_sh', 'mde_80pct_diff_sh')
    for col in pct:
        P[col] = 100 * P[col]
    P = P.rename(columns={c: c + '_pct_mo' for c in pct})
    save('T5_power', P.set_index(['country', 'factor']))

    # --- components of the standard errors (Table B1) -----------------------
    b1 = []
    for c, r in head.items():
        d = standard_error_components(r, overlap_lags=HEADLINE - 1)
        d.insert(0, 'country', c)
        b1.append(d)
    save('B1_standard_errors', pd.concat(b1, ignore_index=True).set_index(['country', 'factor']))

    # --- robustness grid (Tables 3 and B2) and lambda / E[f] by window -------
    # Three beta windows times three settings: value-weighted legs with JKP
    # factors (baseline), equal-weighted legs, French Europe factors.
    grid, ratios = [], []
    for c in countries:
        for w in (36, 60, 120):
            for wt in ('vw_cap', 'ew'):
                for fsrc in ('jkp', 'french'):
                    if wt == 'ew' and fsrc == 'french':
                        continue
                    r = baseline.run_baseline(c, window=w, weighting=wt,
                                              factor_source=fsrc, verbose=False)
                    sh = r['shanken'].set_index('factor')
                    row = dict(country=c, window=w, weighting=wt, factors=fsrc,
                               T=r['gamma0']['T'], grs=r['grs']['grs'],
                               grs_p=r['grs']['p_value'],
                               wald_p=r['wald_ef'].get('p_value_nw_sh'),
                               gamma0_t_nw=r['gamma0']['t_nw_sh'])
                    ed = r['lam_vs_ef'].set_index('factor')
                    for f in r['factor_cols']:
                        row['t_' + f] = sh.loc[f, 't_nw_shanken_ef']
                        row['tdiff_' + f] = ed.loc[f, 't_diff_nw_sh']
                    grid.append(row)
                    if wt == 'vw_cap' and fsrc == 'jkp':
                        lam = r['lambdas']
                        ef = r['factors'].loc[r['factors'].index.intersection(lam.index)]
                        for f in r['factor_cols']:
                            ratios.append(dict(country=c, window=w, factor=f,
                                               ratio=float(lam['lambda_' + f].mean()
                                                           / ef[f].mean())))
                    del r, sh
                    gc.collect()
                    print('   grid %s w=%-3d %-6s %-6s' % (c, w, wt, fsrc), flush=True)
    save('T7_robustness_grid', pd.DataFrame(grid).set_index(
        ['country', 'window', 'weighting', 'factors']))
    save('premia_ratio_by_window', pd.DataFrame(ratios).pivot_table(
        index=['country', 'factor'], columns='window', values='ratio'))
    print('-> %s' % config.OUTPUT)


if __name__ == '__main__':
    main()
