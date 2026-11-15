"""Appendix C: the cap of eight factors.

The reliability of the estimated betas is the share of their cross-sectional
variation that is not estimation noise. It is measured by the correlation of
betas estimated on the odd and the even months of each 60-month window,
stepped up with the Spearman-Brown formula 2r / (1 + r). For each K, the
model is the market plus K - 1 random candidates (12 draws per K). The cap is
the largest K at which the mean reliability stays at or above 80 percent of
its level at K = 6, in Germany and the United Kingdom.

Writes k_cap.csv.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np                                      # noqa: E402
import pandas as pd                                     # noqa: E402

from eufactors import baseline, config, data, fmb      # noqa: E402

KS = [6, 7, 8, 9, 10, 11, 12, 14]
DRAWS = 12
WINDOW = 60
SEED = 2026
MARKETS = ('DEU', 'GBR')


def candidates(country):
    """Non-test-asset factors with at least 300 months since January 1994."""
    w = data.factors(country)
    w = w[[c for c in w.columns if c not in baseline.CHARS]].loc['1994-01':]
    return w.dropna(axis=1, thresh=300)


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for country in MARKETS:
        legs = baseline.leg_panel(country)
        Z = candidates(country)
        mkt = data.classic_factors(country.lower())[['MKT']]
        for k in KS:
            vals = []
            for _ in range(DRAWS):
                pick = list(rng.choice(list(Z.columns), size=k - 1, replace=False))
                F = mkt.join(Z[pick], how='inner').dropna().loc['1994-01-01':]
                panel = data.merge_monthly(legs, F)
                bs = fmb.rolling_betas_split(panel, list(F.columns), window=WINDOW,
                                             min_periods=int(WINDOW * 2 / 3))
                rr = fmb.split_half_correlation(bs, list(F.columns))
                vals.append(float((2 * rr / (1 + rr)).mean()))
            v = np.array(vals)
            rows.append(dict(country=country, K=k, mean=v.mean(), sd=v.std(ddof=1),
                             lo=np.percentile(v, 10), hi=np.percentile(v, 90)))
            print('  %s K=%-3d %.3f  (sd %.3f)' % (country, k, v.mean(), v.std(ddof=1)), flush=True)

    d = pd.DataFrame(rows)
    out = []
    for c, g in d.groupby('country'):
        base = float(g.loc[g.K == 6, 'mean'].iloc[0])
        out.append(g.assign(share_of_K6=g['mean'] / base))
    d = pd.concat(out)
    d.to_csv(config.out('k_cap.csv'), index=False)
    print(d.pivot_table(index='K', columns='country', values='share_of_K6').round(2).to_string())


if __name__ == '__main__':
    main()
