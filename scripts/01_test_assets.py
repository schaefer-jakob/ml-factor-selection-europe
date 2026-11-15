"""Section 3.2: eligibility of the 153 JKP characteristics as test assets.

Conditions, measured in the reference markets Germany and the United Kingdom:
  1. the largest |corr| of the characteristic's factor with SMB, HML, RMW,
     CMA or WML is at most 0.55 (larger of the two markets);
  2. its legs have at least 300 months of returns (smaller of the two);
  3. the three legs together hold at least 150 stocks on average (smaller
     of the two);
  4. returns start no later than December 1994;
  5. all three legs have a return in at least 95 percent of the months from
     January 1994 (both markets).
The five classic characteristics are excluded. Conditions 1 to 3 leave 69
characteristics, all five leave 56.

Writes test_asset_candidates.csv (conditions 1 to 3) and
selected_characteristics.csv (all five) to the output directory. The copies
in config/ are the ones the other scripts use.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402

from eufactors import baseline, config, data       # noqa: E402

CLASSIC = list(data.CLASSICS)
STYLES = ['SMB', 'HML', 'RMW', 'CMA', 'WML']
REFERENCE = ('deu', 'gbr')
START = '1994-01-01'


def legs(country, char):
    path = baseline.leg_path(country, char)
    if not os.path.exists(path):
        return None, None
    df = baseline._read_zip(path)
    df['date'] = pd.to_datetime(df['date'])
    return df.pivot(index='date', columns='pf', values='ret'), df


def ols(y, X):
    X = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X, y, rcond=None)[0][1:]


def screen(country):
    F = data.factors(country)
    rhs = pd.concat([data.market(country), F[CLASSIC]], axis=1).dropna()
    rhs.columns = ['MKT'] + STYLES
    out = []
    for char in sorted(F.columns):
        L, raw = legs(country, char)
        if L is None or L.shape[1] < 3:
            continue
        J = L.join(rhs, how='inner').dropna()
        if len(J) < 180:
            continue
        X = J[rhs.columns].to_numpy()
        B = np.array([ols(J[c].to_numpy(), X) for c in L.columns])
        f = F[char].reindex(rhs.index).dropna()
        cc = rhs.loc[f.index].join(f.rename('x')).corr()['x'][STYLES].abs().max()
        out.append(dict(
            char=char, ctry=country,
            maxcorr=cc,
            mkt_spread=B[:, 0].max() - B[:, 0].min(),
            T=len(J), start=J.index.min().strftime('%Y-%m'),
            stocks=raw.groupby('date')['n'].sum().mean()))
    return pd.DataFrame(out).set_index('char')


def complete_share(char):
    """Smallest share, over the reference markets, of months since START in
    which all three legs have a return."""
    out = []
    for c in REFERENCE:
        L = legs(c, char)[0].loc[START:]
        out.append(len(L.dropna()) / max(len(L), 1))
    return min(out)


def main():
    d, g = (screen(c) for c in REFERENCE)
    m = pd.DataFrame({
        'maxcorr': pd.concat([d.maxcorr, g.maxcorr], axis=1).max(axis=1),
        'mkt_spread': pd.concat([d.mkt_spread, g.mkt_spread], axis=1).min(axis=1),
        'T_min': pd.concat([d['T'], g['T']], axis=1).min(axis=1),
        'start': pd.concat([d.start, g.start], axis=1).max(axis=1),
        'stocks': pd.concat([d.stocks, g.stocks], axis=1).min(axis=1),
    }).dropna()
    m = m[~m.index.isin(CLASSIC)]
    elig = m[(m.maxcorr <= 0.55) & (m.T_min >= 300) & (m.stocks >= 150)]
    elig = elig.sort_values('mkt_spread', ascending=False)
    elig.to_csv(config.out('test_asset_candidates.csv'))
    print('conditions 1-3: %d characteristics' % len(elig))

    pool = elig[elig.start <= '1994-12']
    pool = pool[[complete_share(ch) >= 0.95 for ch in pool.index]]
    selected = sorted(pool.index)
    pd.Series(selected).to_csv(config.out('selected_characteristics.csv'), index=False,
                               header=['characteristic'])
    print('conditions 1-5: %d characteristics, %d test assets per market'
          % (len(selected), 3 * len(selected)))
    same = selected == sorted(baseline.CHARS)
    print('identical to config/selected_characteristics.csv: %s' % same)


if __name__ == '__main__':
    main()
