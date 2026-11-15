"""Chapter 5: selections by JKP theme (Section 5.2.1 and Appendix C).

For the LASSO and forward stepwise, the share of selections that fall in each
theme is divided by the theme's share among the candidates, so that a method
picking at random has a ratio of one. Two sources: the stability-selection
frequencies (P1) and the 26 rolling windows (P11_window_sets). Themes are the
cluster labels of Jensen et al. (2023), config/jkp_cluster_labels.csv.

Writes I5_theme_shares.csv.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import pandas as pd                     # noqa: E402

from eufactors import config            # noqa: E402

METHODS = ['LASSO', 'ForwardStepwise']


def main():
    lab = pd.read_csv(config.config('jkp_cluster_labels.csv')).set_index('characteristic').cluster
    freq = pd.read_csv(config.out('P1_stability_frequencies.csv'), index_col=0)
    win = pd.read_csv(config.out('P11_window_sets.csv'))
    cands = freq.index
    theme = lab.reindex(cands)
    if theme.isna().any():
        raise ValueError('candidates without a theme: %s' % list(cands[theme.isna()]))
    n = len(cands)
    cand_share = theme.value_counts() / n
    rows = []
    for c in config.COUNTRIES:
        for m in METHODS:
            s = freq[f'{c} | {m}'].groupby(theme).sum()
            s = s / s.sum()
            for t, v in s.items():
                rows.append(dict(country=c, source='stability_halves', method=m, theme=t,
                                 sel_share=v, cand_share=cand_share[t]))
        for m in METHODS:
            w = win[(win.country == c) & (win.method == m)].members.dropna()
            names = [x for mem in w for x in mem.split('|') if x]
            s = pd.Series([lab[x] for x in names]).value_counts() / len(names)
            for t in cand_share.index:
                rows.append(dict(country=c, source='rolling_windows', method=m, theme=t,
                                 sel_share=s.get(t, 0.0), cand_share=cand_share[t],
                                 n_selections=len(names)))
    df = pd.DataFrame(rows)
    df['ratio'] = df.sel_share / df.cand_share
    df.to_csv(config.out('I5_theme_shares.csv'), index=False)
    pd.set_option('display.width', 200)
    print(df[df.source == 'rolling_windows'].pivot_table(
        index='theme', columns=['method', 'country'], values='ratio').round(2))


if __name__ == '__main__':
    main()
