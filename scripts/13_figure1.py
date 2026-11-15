"""Figure 1: overlap of selected factor sets by separation between windows.

For each market and method, the ratio of the mean Jaccard index of the sets
of windows w and w + s to that of random sets of the same sizes
(P2c_by_separation.csv). The band is the 5th to 95th percentile of the same
ratio when the LASSO set of every window is replaced by a random set of the
same size, 2,000 draws.

Reads P11_window_sets.csv and P2c_by_separation.csv and writes
P2e_overlap_placebo_band.csv and figure1_overlap.png. Requires matplotlib.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import matplotlib                                    # noqa: E402
import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402

from eufactors import config                         # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib import font_manager as fm            # noqa: E402

DRAWS, SEED = 2000, 20261101
METHODS = [('ForwardStepwise', 'Forward stepwise', '#000000', '-', 'o'),
           ('LASSO', 'LASSO', '#0072B2', '-', 's'),
           ('ElasticNet', 'Elastic net', '#D55E00', '--', '^'),
           ('LASSO CV', 'LASSO CV', '#009E73', ':', 'D')]
SEPS = (1, 2, 3, 5, 6, 10)
NAMES = {'DEU': 'Germany', 'GBR': 'United Kingdom', 'FRA': 'France'}
FLOOR = 0.15          # zero overlap is drawn on the bottom line of the log axis


def jac(a, b):
    u = a | b
    return len(a & b) / len(u) if u else np.nan


def main():
    W = pd.read_csv(config.out('P11_window_sets.csv'))
    C = pd.read_csv(config.out('P2c_by_separation.csv'))
    P = len(pd.read_csv(config.out('P1_stability_frequencies.csv'), index_col=0))

    def windows(c, m):
        d = W[(W.country == c) & (W.method == m)].sort_values('window')
        return [set(x.split('|')) if isinstance(x, str) and x else set() for x in d.members]

    sel_methods = [x[0] for x in METHODS]
    # the window sets must reproduce the observed Jaccard index of P2c
    for (c, m), g in C[C.method.isin(sel_methods)].groupby(['country', 'method']):
        per = windows(c, m)
        for _, r in g.iterrows():
            s = int(r.separation)
            obs = np.nanmean([jac(per[i], per[i + s]) for i in range(len(per) - s)])
            assert abs(obs - r.observed) < 1e-9, (c, m, s, obs, r.observed)

    rng = np.random.default_rng(SEED)
    rows = []
    for (c, m), g in C[C.method.isin(sel_methods)].groupby(['country', 'method']):
        sizes = [len(x) for x in windows(c, m)]
        sims = {s: [] for s in SEPS}
        for _ in range(DRAWS):
            rnd = [set(rng.choice(P, k, replace=False).tolist()) for k in sizes]
            for s in SEPS:
                sims[s].append(np.nanmean([jac(rnd[i], rnd[i + s]) for i in range(len(rnd) - s)]))
        for _, r in g.iterrows():
            s = int(r.separation)
            x = np.array(sims[s]) / r.random
            rows.append(dict(country=c, method=m, separation=s, ratio=r.ratio,
                             band_lo=np.nanpercentile(x, 5), band_hi=np.nanpercentile(x, 95),
                             p_below=np.nanmean(x <= r.ratio)))
    B = pd.DataFrame(rows)
    B.to_csv(config.out('P2e_overlap_placebo_band.csv'), index=False)

    fam = [f.name for f in fm.fontManager.ttflist]
    plt.rcParams.update({'font.family': 'Palatino Linotype' if 'Palatino Linotype' in fam else 'serif',
                         'font.size': 8, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                         'ytick.major.width': 0.6, 'axes.spines.top': False,
                         'axes.spines.right': False, 'mathtext.fontset': 'cm'})
    markets = [c for c in NAMES if c in set(C.country)]
    fig, axs = plt.subplots(1, len(markets), figsize=(6.27, 2.45), sharey=True, squeeze=False)
    axs = axs[0]
    for k, (ax, c) in enumerate(zip(axs, markets)):
        ax.axvspan(5.5, 10.5, color='0.92', lw=0, zorder=0)
        # the penalized methods have the same set sizes, so one band serves all of them
        b = B[(B.country == c) & (B.method == 'LASSO')].sort_values('separation')
        ax.fill_between(b.separation, b.band_lo.clip(lower=FLOOR), b.band_hi, color='#0072B2',
                        alpha=0.13, lw=0, zorder=1, label='90% range of random sets (LASSO)')
        ax.axhline(1, color='0.4', lw=0.7, zorder=1)
        for m, lab, col, ls, mk in METHODS:
            d = C[(C.country == c) & (C.method == m)].sort_values('separation')
            ax.plot(d.separation, d.ratio.clip(lower=FLOOR), color=col, ls=ls, lw=1.1, marker=mk,
                    ms=3.2, label=lab, zorder=3)
            z = d[d.ratio <= 0]
            ax.plot(z.separation, [FLOOR] * len(z), ls='none', marker=mk, ms=3.6, mfc='white',
                    mec=col, mew=0.9, zorder=4)
        ax.set_yscale('log')
        ax.set_ylim(FLOOR * 0.85, 30)
        ax.set_yticks([FLOOR, 0.25, 0.5, 1, 2, 5, 10, 20])
        ax.set_yticklabels(['0', '0.25', '0.5', '1', '2', '5', '10', '20'])
        ax.minorticks_off()
        ax.set_xticks([1, 2, 3, 5, 6, 10])
        ax.set_xlim(0.5, 10.5)
        ax.set_xlabel('Separation $s$ (windows)')
        ax.set_title(f'({"ABC"[k]}) {NAMES[c]}', fontsize=8.5, loc='left')
        ax.text(8, 22, 'no shared\nmonths', ha='center', va='top', fontsize=7, color='0.35')
    axs[0].set_ylabel('Overlap relative to chance')
    h, lab = axs[0].get_legend_handles_labels()
    order = [1, 2, 3, 4, 0]
    fig.legend([h[i] for i in order], [lab[i] for i in order], loc='lower center', ncol=5,
               frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.01), handlelength=1.8,
               columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.8)
    fig.savefig(config.out('figure1_overlap.png'), dpi=600)
    print(B[B.separation.isin([1, 6])].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
