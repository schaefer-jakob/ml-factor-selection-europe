"""Local-currency versions of the JKP US-dollar excess returns (Section 3.1,
Table B5).

JKP measure every return in US dollars in excess of the US one-month
T-bill. With s the monthly appreciation of the local currency against the
dollar and x = s + rf_loc - rf_US the currency excess return,

    long-only portfolio and MKT:  r_usd ~ r_loc + x + R_loc * s
    long-short factor:            f_usd = f_loc * (1 + s)   (exact)

This module inverts the conversion:

    long-only:   1 + R_loc = (1 + r_usd + rf_US) / (1 + s),   r_loc = R_loc - rf_loc
    long-short:  f_loc = f_usd / (1 + s)

Inputs:
  * Exchange rates from 1998: Compustat Global daily rates (USD per unit of
    local currency), last available day of each month. Germany and France use
    DEM and FRF before 1999 and EUR afterwards, scaled by the fixed conversion
    rates.
  * Exchange rates 1994-1997: FRED. GBP, CHF and SEK are daily noon rates
    (DEXUSUK, DEXSZUS, DEXSDUS), taken at month-end. For DEM and FRF only
    monthly averages are available (EXGEUS, EXFRUS).
  * US rate: the RF column of the French data library.
  * Local rate: OECD three-month interbank rates from FRED (IR3TIB01xxM156N,
    percent per year), converted to a monthly rate and lagged by one month.
    Switzerland uses the call-money rate (IRSTCI01CHM156N) before the
    interbank series starts in July 1999.
"""
from functools import lru_cache

import pandas as pd

from . import config, data

COMPUSTAT_START = '1998-01'

# country -> (currency before 1999, currency from 1999, FRED series 1994-1997,
#             FRED quoting 'usd_per' or 'per_usd', FRED series is daily)
CURRENCY = {
    'deu': ('DEM', 'EUR', 'EXGEUS', 'per_usd', False),
    'fra': ('FRF', 'EUR', 'EXFRUS', 'per_usd', False),
    'gbr': ('GBP', 'GBP', 'DEXUSUK', 'usd_per', True),
    'che': ('CHF', 'CHF', 'DEXSZUS', 'per_usd', True),
    'swe': ('SEK', 'SEK', 'DEXSDUS', 'per_usd', True),
}
LOCAL_RATE = {
    'deu': ['IR3TIB01DEM156N'],
    'fra': ['IR3TIB01FRM156N'],
    'gbr': ['IR3TIB01GBM156N'],
    'che': ['IR3TIB01CHM156N', 'IRSTCI01CHM156N'],
    'swe': ['IR3TIB01SEM156N'],
}
EURO_RATES = {'DEM': 1.95583, 'FRF': 6.55957}


def _fred(series_id):
    df = pd.read_csv(config.data_path('fred', series=series_id))
    df.columns = ['date', 'value']
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df.dropna().set_index('date')['value']


@lru_cache(maxsize=1)
def _compustat_month_end():
    """USD per unit of each currency, last available day of each month."""
    d = pd.read_csv(config.data_path('compustat_fx'),
                    usecols=['curd', 'datadate', 'exratd_tousd'])
    d['datadate'] = pd.to_datetime(d['datadate'])
    d = d.dropna(subset=['exratd_tousd'])
    # several rate types per day carry the same cross rate; take the median
    d = d.groupby(['curd', 'datadate'])['exratd_tousd'].median().reset_index()
    d['ym'] = d['datadate'].dt.to_period('M')
    last = d.sort_values('datadate').groupby(['curd', 'ym']).tail(1)
    return last.pivot(index='ym', columns='curd', values='exratd_tousd')


def usd_per_local(country):
    """Month-end level in USD per unit of local currency, monthly PeriodIndex."""
    pre, post, fred_id, quote, daily = CURRENCY[country.lower()]
    cs = _compustat_month_end()
    level = cs[post].copy()
    if pre != post:
        early = cs[pre] * EURO_RATES[pre]
        level = level.where(level.index >= pd.Period('1999-01', 'M'), early)
    f = _fred(fred_id)
    if daily:
        f = f.groupby(f.index.to_period('M')).last()
    else:
        f.index = f.index.to_period('M')            # monthly average
    f = f if quote == 'usd_per' else 1.0 / f
    if pre != post:
        f = f * EURO_RATES[pre]
    early = f[f.index < pd.Period(COMPUSTAT_START, 'M')]
    return pd.concat([early, level[level.index >= pd.Period(COMPUSTAT_START, 'M')]]).sort_index()


def fx_return(country):
    """s_t: monthly appreciation of the local currency against the dollar."""
    return usd_per_local(country).pct_change().dropna().rename('s')


def _monthly_rate(pct_pa):
    return (1.0 + pct_pa / 100.0) ** (1.0 / 12.0) - 1.0


def local_rf(country):
    """Monthly local risk-free rate for month t, lagged by one month."""
    ids = LOCAL_RATE[country.lower()]
    r = _fred(ids[0])
    for alt in ids[1:]:
        r = r.combine_first(_fred(alt))
    r.index = r.index.to_period('M')
    return _monthly_rate(r).shift(1).dropna().rename('rf_loc')


def us_rf():
    rf = data.french('FF5')['RF']
    rf.index = rf.index.to_period('M')
    return rf.rename('rf_us')


def conversion_table(country):
    """One row per month: s, rf_us, rf_loc and the currency excess return x."""
    t = pd.concat([fx_return(country), us_rf(), local_rf(country)], axis=1).dropna()
    t['x'] = t['s'] + t['rf_loc'] - t['rf_us']
    return t


def long_only_to_local(r_usd_excess, months, country):
    """USD excess return of a long-only portfolio -> local excess return."""
    t = conversion_table(country).reindex(months)
    R_loc = (1.0 + r_usd_excess + t['rf_us'].to_numpy()) / (1.0 + t['s'].to_numpy()) - 1.0
    return R_loc - t['rf_loc'].to_numpy()


def long_short_to_local(f_usd, months, country):
    """USD long-short return -> local long-short return."""
    t = conversion_table(country).reindex(months)
    return f_usd / (1.0 + t['s'].to_numpy())


def legs_to_local(legs, country):
    """baseline.leg_panel output in local currency. The return column keeps
    its name so that the estimation code runs unchanged."""
    out = legs.copy()
    months = pd.PeriodIndex(out['datadate'], freq='M')
    out['ret_usd_excess'] = long_only_to_local(out['ret_usd_excess'].to_numpy(), months, country)
    return out.dropna(subset=['ret_usd_excess'])


def factors_to_local(cf, country, long_only=('MKT',)):
    """data.classic_factors output in local currency: MKT as a long-only
    excess return, every other column as a long-short return."""
    out = cf.copy()
    months = pd.PeriodIndex(out.index, freq='M')
    for c in out.columns:
        conv = long_only_to_local if c in long_only else long_short_to_local
        out[c] = conv(out[c].to_numpy(), months, country)
    return out.dropna(how='all')
