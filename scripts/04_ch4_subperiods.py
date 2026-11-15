"""Chapter 4: change of the gaps lambda_t - f_t between sub-periods (Section
4.1.3, Table B6 and the last row of Table 3).

The sample is split at the end of 2008 (May 1997 to December 2008 and January
2009 to December 2025). Per factor, the mean gap in each period has a
Newey-West t-statistic, and the change is tested with

    t = (mean_2 - mean_1) / sqrt(se_1^2 + se_2^2).

The joint Wald test of no change uses the sum of the two Newey-West
covariance matrices and is chi2(6) under H0. No Shanken factor is applied.

Writes C1_lambda_gap_subperiods.csv and C1b_lambda_gap_change_wald.csv.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np                              # noqa: E402
import pandas as pd                             # noqa: E402

from eufactors import baseline, config, fmb     # noqa: E402

BREAK = '2008-12-31'


def main():
    rows, model = [], []
    for c in config.COUNTRIES:
        res = baseline.run_baseline(c, window=60, verbose=True)
        lam, cf, fac = res['lambdas'], res['factors'], res['factor_cols']
        idx = lam.index.intersection(cf.index)
        D = pd.DataFrame({f: lam.loc[idx, 'lambda_' + f] - cf.loc[idx, f] for f in fac})
        p1, p2 = D[D.index <= BREAK], D[D.index > BREAK]
        for f in fac:
            a, b = p1[f].to_numpy(), p2[f].to_numpy()
            s1, s2 = fmb.newey_west_se(a), fmb.newey_west_se(b)
            rows.append(dict(country=c, factor=f, T1=len(a), T2=len(b),
                             d1_pct=100 * a.mean(), d2_pct=100 * b.mean(),
                             t_d1=a.mean() / s1, t_d2=b.mean() / s2,
                             t_change=(b.mean() - a.mean()) / np.sqrt(s1 ** 2 + s2 ** 2)))
        V = fmb.newey_west_cov(p1.to_numpy()) + fmb.newey_west_cov(p2.to_numpy())
        g = (p2.mean() - p1.mean()).to_numpy()
        W = float(g @ np.linalg.solve(V, g))
        model.append(dict(country=c, wald_change=W, df=len(fac),
                          p_value=fmb.chi2_sf(W, len(fac))))
    T = pd.DataFrame(rows).set_index(['country', 'factor'])
    M = pd.DataFrame(model).set_index('country')
    T.to_csv(config.out('C1_lambda_gap_subperiods.csv'))
    M.to_csv(config.out('C1b_lambda_gap_change_wald.csv'))
    print(T.round(3).to_string())
    print(M.round(4).to_string())


if __name__ == '__main__':
    main()
