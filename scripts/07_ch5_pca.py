"""Chapter 5: selection in principal-component space and subspace alignment.

In each rolling window the LASSO (q = 8) selects among the first 30
principal components of the candidates. The subspaces of the selected
components in windows w and w + s are compared by their alignment, against
random draws of the same number of components from each window's first 30,
and against PC1-PC8 without selection.

Writes P8_pca_subspace.csv (all window pairs) and P8b_pca_by_separation.csv
(by separation; the random-subspace row of Table C3, Panel B).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402

from eufactors import config, pca, rolling           # noqa: E402
from eufactors import selection as sel               # noqa: E402

N_TOP = 30


def main():
    U = sel.shared_universe_cut(config.COUNTRIES)
    print('candidate set: %d factors' % len(U), flush=True)
    rows_pc, rows_sep = [], []

    for c in config.COUNTRIES:
        R, m, Z = sel.sample(c, universe=U, menu_cut=False)
        mk = m.to_numpy().ravel()
        starts = rolling.window_starts(len(R))

        loads, ks, shares, tops = [], [], [], []
        for s0 in starts:
            sl = slice(s0, s0 + sel.WINDOW)
            V, S, share = pca.pcs(Z.iloc[sl])
            n_cand = min(N_TOP, V.shape[1])
            tops.append(V[:, :n_cand])
            Sdf = pd.DataFrame(S[:, :n_cand], index=Z.index[sl],
                               columns=['PC%02d' % (i + 1) for i in range(n_cand)])
            pick, _ = sel.penalised_select(R.iloc[sl], mk[sl], Sdf, alpha=1.0, cap=sel.CAP)
            ix = [int(x[2:]) - 1 for x in pick]
            loads.append(V[:, ix])
            ks.append(len(ix))
            shares.append(share[ix].sum())

        cons = [pca.alignment(loads[i], loads[i + 1]) for i in range(len(loads) - 1)]
        allp = [pca.alignment(loads[i], loads[j])
                for i in range(len(loads)) for j in range(i + 1, len(loads))]
        rng = np.random.default_rng(11)
        rnd = np.nanmean([pca.subspace_null(tops[i], tops[j],
                                            loads[i].shape[1], loads[j].shape[1], rng, n=60)
                          for i in range(len(loads)) for j in range(i + 1, len(loads))])
        base = np.nanmean([pca.alignment(tops[i][:, :sel.CAP], tops[j][:, :sel.CAP])
                           for i in range(len(tops)) for j in range(i + 1, len(tops))])
        for lag in (1, 3, 6):
            idx = [(i, i + lag) for i in range(len(loads) - lag)]
            obs_l = np.nanmean([pca.alignment(loads[i], loads[j]) for i, j in idx])
            nul_l = np.nanmean([pca.subspace_null(tops[i], tops[j], loads[i].shape[1],
                                                  loads[j].shape[1], rng, n=60)
                                for i, j in idx])
            base_l = np.nanmean([pca.alignment(tops[i][:, :sel.CAP], tops[j][:, :sel.CAP])
                                 for i, j in idx])
            rows_sep.append(dict(country=c, separation=lag,
                                 months_shared=max(0, sel.WINDOW - lag * rolling.STEP),
                                 n_pairs=len(idx), aligned_selected=obs_l,
                                 aligned_random=nul_l, ratio=obs_l / nul_l,
                                 aligned_no_selection=base_l))
        rows_pc.append(dict(country=c, n_windows=len(starts), mean_K=np.mean(ks),
                            mean_var_share=np.mean(shares),
                            align_consecutive=np.nanmean(cons),
                            align_all_pairs=np.nanmean(allp),
                            align_random=rnd,
                            ratio_vs_random=np.nanmean(allp) / rnd,
                            no_selection_PC1toK=base,
                            selection_beats_no_selection=np.nanmean(allp) > base))
        print('  %s PC: K=%.1f, alignment %.3f | null %.3f | PC1-PC8 %.3f'
              % (c, np.mean(ks), np.nanmean(allp), rnd, base), flush=True)

    pd.DataFrame(rows_pc).set_index('country').to_csv(config.out('P8_pca_subspace.csv'))
    pd.DataFrame(rows_sep).set_index(['country', 'separation']).to_csv(
        config.out('P8b_pca_by_separation.csv'))
    print('-> %s' % config.OUTPUT)


if __name__ == '__main__':
    main()
