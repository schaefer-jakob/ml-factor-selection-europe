"""Readers for the JKP Global Factor Data and the Kenneth French Data Library.

All JKP returns are monthly US-dollar excess returns of capped value-weighted
portfolios, as published (Section 3.1). Every function here is read-only.
"""
import io
import zipfile

import pandas as pd

from . import config

#: JKP characteristic of each classic factor, and its Fama-French name.
CLASSICS = {
    'market_equity': 'SMB',
    'be_me': 'HML',
    'ope_be': 'RMW',
    'at_gr1': 'CMA',
    'ret_12_1': 'WML',
}

#: Pseudo-countries that pool several JKP markets (Section 5.1.5). The test
#: assets of the members are stacked into one cross-section, and every factor,
#: the market included, is the equal-weighted average of the members' returns.
POOLS = {'chse': ('che', 'swe')}


def read_zip(path):
    """First CSV inside a zip archive."""
    with zipfile.ZipFile(path) as z:
        return pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))


def _pool(frames, keys):
    """Equal-weighted average of the members' `ret`, only in months in which
    every member has a value."""
    m = frames[0][keys + ['ret']].dropna(subset=['ret'])
    for i, f in enumerate(frames[1:], start=1):
        m = m.merge(f[keys + ['ret']].dropna(subset=['ret'])
                    .rename(columns={'ret': 'ret_%d' % i}), on=keys, how='inner')
    cols = ['ret'] + ['ret_%d' % i for i in range(1, len(frames))]
    m['ret'] = m[cols].mean(axis=1)
    return m[keys + ['ret']]


def factors_raw(country):
    """All 153 JKP long-short factors of one market, long format."""
    if country.lower() in POOLS:
        return _pool([factors_raw(c) for c in POOLS[country.lower()]], ['date', 'name'])
    return read_zip(config.data_path('jkp_factors', cty=country.lower()))


def factors(country):
    """All 153 factors, wide: index = month-end date, columns = factor name."""
    df = factors_raw(country)
    df['date'] = pd.to_datetime(df['date'])
    return df.pivot(index='date', columns='name', values='ret')


def market(country):
    """JKP market factor (excess return) as a Series named 'mkt'."""
    if country.lower() in POOLS:
        df = _pool([read_zip(config.data_path('jkp_market', cty=c))
                    for c in POOLS[country.lower()]], ['date'])
    else:
        df = read_zip(config.data_path('jkp_market', cty=country.lower()))
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')['ret'].rename('mkt')


def classic_factors(country):
    """The six factors of the baseline model from the JKP country data
    (Section 3.1): MKT plus the long-short factors of the five classic
    characteristics, named SMB, HML, RMW, CMA and WML."""
    f = factors(country)[list(CLASSICS.keys())].copy()
    f.columns = [CLASSICS[c] for c in f.columns]
    f['MKT'] = market(country)
    return f


def french(name):
    """Monthly block of a Kenneth French Europe file, in decimals.

    name: 'FF5' (five factors and RF) or 'MOM' (momentum).
    """
    kind = {'FF5': 'french_ff5', 'MOM': 'french_mom'}[name]
    with zipfile.ZipFile(config.data_path(kind)) as z:
        text = z.read(z.namelist()[0]).decode('latin-1')

    rows, header = [], None
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if header is None:
            # header row: first cell empty, the rest are column names
            if parts[0] == '' and len(parts) > 1 and any(parts[1:]):
                header = parts[1:]
            continue
        # monthly block: six-digit YYYYMM key; stop at the first other row
        if len(parts) == len(header) + 1 and parts[0].isdigit() and len(parts[0]) == 6:
            rows.append(parts)
        elif rows:
            break

    df = pd.DataFrame(rows, columns=['date'] + header)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m') + pd.offsets.MonthEnd(0)
    df = df.set_index('date').apply(pd.to_numeric, errors='coerce')
    return df / 100.0  # the library reports percent


def french_classic_factors():
    """French Europe five factors and momentum, with the same column names as
    classic_factors(). The factors are regional, so every market uses the same
    panel (Section 3.3 and the factor-source robustness check)."""
    ff5 = french('FF5')
    mom = french('MOM')
    f = ff5[['SMB', 'HML', 'RMW', 'CMA']].join(mom[['WML']], how='outer')
    f['MKT'] = ff5['Mkt-RF']
    return f


def merge_monthly(df, monthly, date_col='datadate'):
    """Merge a panel onto a month-indexed frame or series by calendar month
    rather than by exact date."""
    if isinstance(monthly, pd.Series):
        monthly = monthly.to_frame()
    left = df.copy()
    left['_ym'] = left[date_col].dt.to_period('M')
    right = monthly.copy()
    right['_ym'] = right.index.to_period('M')
    out = left.merge(right, on='_ym', how='left')
    return out.drop(columns='_ym')
