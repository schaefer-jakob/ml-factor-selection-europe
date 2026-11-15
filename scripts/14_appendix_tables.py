"""Appendix A tables and the candidate-correlation check of Appendix C.

  A1_test_assets_by_theme.csv     test-asset characteristics by JKP theme (Table A1)
  A2_jkp_vs_french.csv            JKP country factors against the French Europe
                                  factors, January 1994 to December 2025 (Table A2)
  A3_candidates_by_theme.csv      the 65 candidates by theme, with the four
                                  removed twins and the classic characteristics
                                  marked (Table A3; Section 5.1.1 theme counts)
  C0_candidate_max_corr.csv       per candidate and market, the largest absolute
                                  correlation with a test-asset characteristic's
                                  factor on the balanced panel, and whether it
                                  exceeds 0.70 (Appendix C, rejected
                                  correlation threshold)

Run in the main scope.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np                                      # noqa: E402
import pandas as pd                                     # noqa: E402

from eufactors import baseline, config, data            # noqa: E402
from eufactors import selection as sel                  # noqa: E402

START, END = '1994-01-01', '2025-12-31'
CORR_THRESHOLD = 0.70


def themes():
    return pd.read_csv(config.config('jkp_cluster_labels.csv')).set_index('characteristic')['cluster']


def by_theme(chars, lab):
    t = pd.DataFrame({'characteristic': chars, 'theme': [lab.get(c) for c in chars]})
    g = t.groupby('theme')['characteristic'].agg(['size', lambda s: ', '.join(sorted(s))])
    g.columns = ['number', 'characteristics']
    return g.sort_values('number', ascending=False)


def jkp_vs_french(countries):
    french = data.french_classic_factors().loc[START:END]
    cols = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'WML']
    rows = []
    for f in cols:
        row = dict(factor=f, french_mean_pct=100 * french[f].mean(),
                   french_sd_pct=100 * french[f].std(), french_n=int(french[f].notna().sum()))
        for c in countries:
            j = data.classic_factors(c.lower()).loc[START:END, f]
            both = pd.concat([j, french[f]], axis=1, keys=['jkp', 'french']).dropna()
            row['%s_corr' % c] = float(both['jkp'].corr(both['french']))
            row['%s_mean_pct' % c] = 100 * j.mean()
            row['%s_sd_pct' % c] = 100 * j.std()
            row['%s_n' % c] = int(len(both))
        rows.append(row)
    return pd.DataFrame(rows).set_index('factor')


def candidates_by_theme(countries, lab):
    u = sel.shared_universe(tuple(countries))
    excluded = set(sel.excluded_candidates())
    t = pd.DataFrame({'candidate': u})
    t['theme'] = [lab.get(c) for c in u]
    t['classic'] = t['candidate'].isin(list(data.CLASSICS))
    t['excluded_twin'] = t['candidate'].isin(excluded)
    return t.sort_values(['excluded_twin', 'theme', 'candidate']).set_index('candidate')


def max_corr_with_test_assets(countries):
    """Largest |corr| of every candidate with the factor of any test-asset
    characteristic, on each market's balanced panel."""
    u = sel.shared_universe(tuple(countries))
    out = {}
    for c in countries:
        _, _, Z = sel.sample(c, universe=u, menu_cut=False)
        w = data.factors(c)
        w.index = pd.PeriodIndex(w.index, freq='M')
        ta = [x for x in baseline.CHARS if x in w.columns]
        B = w.loc[w.index.isin(Z.index), ta].dropna(axis=1)
        common = Z.index.intersection(B.index)
        A = Z.loc[common]
        C = np.corrcoef(A.to_numpy().T, B.loc[common].to_numpy().T)
        out[c] = pd.Series(np.abs(C[:A.shape[1], A.shape[1]:]).max(axis=1), index=A.columns)
    M = pd.DataFrame(out)
    for c in countries:
        M['%s_above' % c] = M[c] > CORR_THRESHOLD
    M['excluded_twin'] = M.index.isin(sel.excluded_candidates())
    return M.sort_values(countries[-1], ascending=False)


def main():
    countries = list(config.COUNTRIES)
    lab = themes()
    by_theme(baseline.CHARS, lab).to_csv(config.out('A1_test_assets_by_theme.csv'))
    jkp_vs_french(countries).to_csv(config.out('A2_jkp_vs_french.csv'))
    A3 = candidates_by_theme(countries, lab)
    A3.to_csv(config.out('A3_candidates_by_theme.csv'))
    M = max_corr_with_test_assets(countries)
    M.to_csv(config.out('C0_candidate_max_corr.csv'))
    print(by_theme(list(A3[~A3.excluded_twin].index), lab)[['number']].to_string())
    print(M[M[['%s_above' % c for c in countries]].any(axis=1)].round(2).to_string())


if __name__ == '__main__':
    main()
