"""Chapter 5: placebo p-values of the out-of-sample R^2 (Section 5.1.4,
Table 7, Tables C6 and D2).

The statistic is the mean out-of-sample R^2 over the window pairs of
P11_pairs.csv. Each model is compared with a placebo version of itself:

  selection methods  one random permutation of the candidate labels applied
                     to the whole sequence of selected sets
                     (P11_window_sets.csv); keeps set sizes, overlap across
                     windows and empty windows. 2,000 draws.
  FF5+WML            a random set of five candidates, held fixed across
                     windows. 5,000 draws.
  PC1-PC8            eight random orthonormal directions in the standardized
                     candidate space, held fixed. 5,000 draws.

p = (1 + #{placebo >= observed}) / (M + 1), with its Monte Carlo standard
error; near_005 flags p-values within two standard errors of 0.05.

Reads the outputs of 06_ch5_selection.py (or 16_appendix_bivariate_betas.py)
of the same scope and writes P11p_placebo.csv.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402

from eufactors import config, rolling               # noqa: E402
from eufactors import dense_benchmark as db         # noqa: E402
from eufactors import selection as sel              # noqa: E402

M_FIXED = 5000
M_PERM = 2000
SEED = 20261007


def _r2(Rb, mkb, Zb_cols):
    return rolling.csr2_F(Rb, np.column_stack([mkb, Zb_cols]))


def _p(obs, null):
    null = np.asarray(null)
    M = len(null)
    p = (1 + int((null >= obs - 1e-12).sum())) / (M + 1)
    se = np.sqrt(p * (1 - p) / M)
    return dict(p_placebo=p, mc_se=se, M=M, near_005=bool(abs(p - 0.05) < 2 * se),
                null_mean=float(null.mean()), null_q95=float(np.percentile(null, 95)))


def main():
    rng = np.random.default_rng(SEED)
    countries = list(config.COUNTRIES)
    U = sel.shared_universe_cut(tuple(countries))
    pairs_tab = pd.read_csv(config.out('P11_pairs.csv'))
    sets_tab = pd.read_csv(config.out('P11_window_sets.csv'), keep_default_na=False)
    print('menu %s: %d candidates' % (config.MENU, len(U)), flush=True)
    rows = []
    for c in countries:
        R, m, Z = sel.sample(c, universe=U, menu_cut=False)
        Rv, mk = R.to_numpy(), m.to_numpy().ravel()
        cols = list(Z.columns)
        col_ix = {n: j for j, n in enumerate(cols)}
        Zv = Z.to_numpy()
        p = len(cols)
        st = rolling.window_starts(len(R))
        for sep in rolling.SEPARATIONS_OOS:
            t0 = time.time()
            prs = [(i, slice(st[i], st[i] + sel.WINDOW), slice(st[i + sep], st[i + sep] + sel.WINDOW))
                   for i in range(len(st) - sep)]
            pt = pairs_tab[(pairs_tab.country == c) & (pairs_tab.separation == sep)]
            if len(pt) == 0:
                continue
            Rb = {i: Rv[b_] for i, a_, b_ in prs}
            mb = {i: mk[b_] for i, a_, b_ in prs}
            Zb = {i: Zv[b_] for i, a_, b_ in prs}
            for meth in pt.method.unique():
                obs_rows = pt[pt.method == meth]
                used = sorted(int(w) for w in obs_rows.window)
                if len(used) < 3:
                    continue
                obs = float(obs_rows.R2_out.mean())
                use = [x for x in prs if x[0] in used]
                null = []
                if meth == 'PC1-PC8 (fixed)':
                    kind = 'random dense 8, fixed'
                    for _ in range(M_FIXED):
                        V = db.random_directions(p, sel.CAP, rng)
                        null.append(np.mean([_r2(Rb[i], mb[i], db.project(Z.iloc[a_], Z.iloc[b_], V))
                                             for i, a_, b_ in use]))
                elif meth == 'FF5+MOM (fixed)':
                    kind = 'random named 5, fixed'
                    for _ in range(M_FIXED):
                        S = rng.choice(p, 5, replace=False)
                        null.append(np.mean([_r2(Rb[i], mb[i], Zb[i][:, S]) for i, a_, b_ in use]))
                else:
                    kind = 'label permutation'
                    ws = sets_tab[(sets_tab.country == c) & (sets_tab.method == meth)]
                    sets = {int(r.window): [col_ix[n] for n in str(r.members).split('|') if n]
                            for r in ws.itertuples()}
                    missing = [i for i, a_, b_ in use if not sets.get(i)]
                    if missing:
                        raise RuntimeError('%s %s s=%d: no stored set for windows %s'
                                           % (c, meth, sep, missing))
                    for _ in range(M_PERM):
                        perm = rng.permutation(p)
                        null.append(np.mean([_r2(Rb[i], mb[i], Zb[i][:, perm[sets[i]]])
                                             for i, a_, b_ in use]))
                rows.append(dict(country=c, separation=sep, method=meth, null=kind,
                                 statistic='mean R2_out', observed=obs, n_pairs=len(use),
                                 **_p(obs, null)))
            print('  %s s=%d  %d methods, %.0fs' % (c, sep, len(pt.method.unique()), time.time() - t0),
                  flush=True)
    out = pd.DataFrame(rows).set_index(['country', 'separation', 'method'])
    out.to_csv(config.out('P11p_placebo.csv'))
    print(out[out.index.get_level_values('separation') == 6][
        ['null', 'observed', 'n_pairs', 'p_placebo', 'mc_se', 'near_005']].round(4).to_string())


if __name__ == '__main__':
    main()
