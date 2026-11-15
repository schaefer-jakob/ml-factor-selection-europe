"""Download the public input files into the data directory.

  JKP Global Factor Data (jkpfactors.com, public files, no login):
    country factors and market factor, monthly, capped value-weighted, for
    DEU, GBR, FRA, CHE and SWE; tercile portfolios of all characteristics in
    DEU and GBR (needed for the test-asset screen), of the 56 test-asset and
    the five classic characteristics in FRA, CHE and SWE, and equal-weighted
    portfolios of the 56 test-asset characteristics in all five markets.
  Kenneth French Data Library: Europe five factors and Europe momentum.
  FRED: exchange rates and interest rates used in fx.py.

Compustat Global exchange rates come from WRDS and are not downloaded here
(see README). Files that already exist are skipped. JKP and French update
their data; the thesis uses the vintage with returns through December 2025.

    python scripts/00_download_data.py
"""
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import pandas as pd                         # noqa: E402

from eufactors import config, data, fx      # noqa: E402

JKP = 'https://jkpfactors-data.s3.amazonaws.com/public/'
FRENCH = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
FRED = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id='
MARKETS = ['deu', 'gbr', 'fra', 'che', 'swe']
FULL_SCREEN = ['deu', 'gbr']
PAUSE = 0.3


def fetch(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return 'present'
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'replication-download/1.0'})
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = r.read()
    except urllib.error.HTTPError as e:
        return 'HTTP %s' % e.code
    except Exception as e:
        return type(e).__name__
    with open(dest, 'wb') as f:
        f.write(payload)
    time.sleep(PAUSE)
    return 'ok'


def jkp_url(*parts, folder=''):
    name = '_'.join('[%s]' % p for p in parts) + '.zip'
    return JKP + folder + urllib.parse.quote(name)


def main():
    failed = []

    def get(url, dest):
        st = fetch(url, dest)
        if st not in ('ok', 'present'):
            failed.append((url, st))
        return st

    for c in MARKETS:
        get(jkp_url(c, 'all_factors', 'monthly', 'vw_cap'), config.data_path('jkp_factors', cty=c))
        get(jkp_url(c, 'mkt', 'monthly', 'vw_cap'), config.data_path('jkp_market', cty=c))
    print('JKP factors done', flush=True)

    test_assets = pd.read_csv(config.config('selected_characteristics.csv'))['characteristic'].tolist()
    for c in MARKETS:
        chars = (sorted(data.factors(c).columns) if c in FULL_SCREEN
                 else sorted(set(test_assets) | set(data.CLASSICS)))
        for ch in chars:
            get(jkp_url(c, ch, 'monthly', 'vw_cap', folder='portfolios/'),
                config.data_path('jkp_portfolios', cty=c, char=ch, weight='vw_cap'))
        for ch in test_assets:
            get(jkp_url(c, ch, 'monthly', 'ew', folder='portfolios/'),
                config.data_path('jkp_portfolios', cty=c, char=ch, weight='ew'))
        print('JKP portfolios %s done' % c, flush=True)

    get(FRENCH + 'Europe_5_Factors_CSV.zip', config.data_path('french_ff5'))
    get(FRENCH + 'Europe_Mom_Factor_CSV.zip', config.data_path('french_mom'))
    series = {v[2] for v in fx.CURRENCY.values()} | {s for v in fx.LOCAL_RATE.values() for s in v}
    for s in sorted(series):
        get(FRED + s, config.data_path('fred', series=s))
    print('French and FRED done', flush=True)

    if not os.path.exists(config.data_path('compustat_fx')):
        print('missing (WRDS, see README): %s' % config.data_path('compustat_fx'))
    for url, st in failed:
        print('FAILED %s -> %s' % (url, st))


if __name__ == '__main__':
    main()
