"""Coverage of the test assets and size of the candidate set by market
(Section 3.2: months with complete legs; Section 5.1.1: shared candidates)."""
import os

import numpy as np
import pandas as pd

from . import baseline
from . import selection as sel

START = '1994-01-01'


def leg_coverage(country):
    """Per test-asset characteristic: file present, share of months since
    START with all three legs, first complete month, smallest leg size."""
    rows = []
    for ch in baseline.CHARS:
        path = baseline.leg_path(country, ch)
        if not os.path.exists(path):
            rows.append(dict(country=country, characteristic=ch, present=False))
            continue
        df = baseline._read_zip(path)
        df['date'] = pd.to_datetime(df['date'])
        L = df.pivot(index='date', columns='pf', values='ret').loc[START:]
        full = L.notna().all(axis=1) & (L.shape[1] == 3)
        rows.append(dict(country=country, characteristic=ch, present=True,
                         n_legs=L.shape[1], months=len(L),
                         complete_share=float(full.mean()) if len(L) else 0.0,
                         first_complete=(L.index[full.values][0].strftime('%Y-%m')
                                         if full.any() else None),
                         min_stocks=(float(df.loc[df['date'] >= START, 'n'].min())
                                     if 'n' in df.columns else np.nan)))
    return pd.DataFrame(rows)


def panel_span(country):
    """Months in which all test-asset legs have a return (the balanced panel
    of Chapter 5)."""
    legs = baseline.leg_panel(country, start=START)
    R = legs.pivot_table(index='datadate', columns='gvkey', values='ret_usd_excess')
    R = R.dropna(axis=1, how='all')
    ok = R.notna().all(axis=1)
    if not ok.any():
        return dict(country=country, n_legs=R.shape[1], T_complete=0)
    idx = R.index[ok.values]
    return dict(country=country, n_legs=R.shape[1], T_complete=int(ok.sum()),
                first=idx.min().strftime('%Y-%m'), last=idx.max().strftime('%Y-%m'),
                gaps=int(ok.loc[idx.min():idx.max()].eq(False).sum()))


def menu_sizes(country_sets):
    """Shared candidate set before and after the menu rule."""
    rows = []
    for cs in country_sets:
        u = sel.shared_universe(tuple(cs), START)
        row = dict(countries='+'.join(cs), shared_before_cut=len(u))
        try:
            row['shared_after_cut'] = len(sel.shared_universe_cut(tuple(cs), START))
        except Exception as e:
            row['shared_after_cut'] = np.nan
            row['note'] = '%s: %s' % (type(e).__name__, e)
        rows.append(row)
    return pd.DataFrame(rows)
