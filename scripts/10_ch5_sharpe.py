"""Chapter 5: out-of-sample Sharpe ratio of the stochastic discount factor
(Section 5.1.4, Tables 6, 7 and C4).

Following Didisheim et al. (2025, p. 17), a model is judged by the
out-of-sample Sharpe ratio of the mean-variance portfolio of its factors.

  * Factor sets: the per-window sets of 06_ch5_selection.py and of the
    lucky-factors bootstrap (08_ch5_benchmarks.py), the six-factor model and
    PC1-PC8. The market is part of every model.
  * Weights are estimated in window w (72 months): tangency Sigma^-1 mu,
    ridge (standardized factors, Sigma + I) and equal weights.
  * They are applied to the 12 months 12(s-1)+1 to 12s after the end of w,
    which w never contains. The blocks are joined across windows into one
    return series per method and separation; its Sharpe ratio is annualized.
  * Placebos as in 09_ch5_placebo.py, run through the same pipeline: label
    permutation of the whole sequence of selected sets, a fixed random set of
    five candidates for the six-factor model, eight fixed random orthonormal
    directions for PC1-PC8. 1,000 draws.

Writes S1_oos_sharpe.csv.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402

from eufactors import config, pca, rolling          # noqa: E402
from eufactors import dense_benchmark as db         # noqa: E402
from eufactors import selection as sel              # noqa: E402

STEP = rolling.STEP
SEPARATIONS = (1, 3, 6)
VARIANTS = ('tangency', 'ridge', 'equal')
B = 1000
CLASSICS = rolling.CLASSICS
K_PC = sel.CAP
NULL_KIND = {'FF5+MOM (fixed)': 'random named 5, fixed',
             'PC1-PC8 (fixed)': 'random dense 8, fixed'}


def weights_from_moments(mu, S, variant):
    """Portfolio weights from the window's mean vector and covariance."""
    k = len(mu)
    if variant == 'equal':
        return np.full(k, 1.0 / k)
    if variant == 'tangency':
        return np.linalg.pinv(S) @ mu
    sd = np.sqrt(np.diag(S))
    return np.linalg.solve(S / np.outer(sd, sd) + np.eye(k), mu / sd) / sd


def sharpe(r):
    sd = r.std(ddof=1)
    return np.sqrt(12.0) * r.mean() / sd if sd > 0 else np.nan


def chained_named(win, idx_of):
    """win: per usable window (mu, S, Xe) over all columns, market first.
    idx_of(i): column indices of the model in window i, or None to skip.
    Returns {variant: joined out-of-sample return series}."""
    out = {v: [] for v in VARIANTS}
    for i, (mu, S, Xe) in win.items():
        ix = idx_of(i)
        if ix is None:
            continue
        m_, S_ = mu[ix], S[np.ix_(ix, ix)]
        for v in VARIANTS:
            out[v].append(Xe[:, ix] @ weights_from_moments(m_, S_, v))
    return {v: (np.concatenate(r) if r else np.array([])) for v, r in out.items()}


def chained_dense(dwin, V_of):
    """dwin: per window (mk_w, Xs_w, mk_e, Xs_e) with standardized candidates.
    V_of(i): p x k loadings. The market is always the first factor."""
    out = {v: [] for v in VARIANTS}
    for i, (mw, Xw, me, Xe) in dwin.items():
        V = V_of(i)
        Fw = np.column_stack([mw, Xw @ V])
        Fe = np.column_stack([me, Xe @ V])
        mu, S = Fw.mean(0), np.cov(Fw, rowvar=False)
        for v in VARIANTS:
            out[v].append(Fe @ weights_from_moments(mu, S, v))
    return {v: np.concatenate(r) for v, r in out.items()}


def main():
    countries = list(config.COUNTRIES)
    U = sel.shared_universe_cut(countries)
    sets_tab = pd.read_csv(config.out('P11_window_sets.csv'), keep_default_na=False)
    lucky_path = config.out('G3b_lucky_window_sets.csv')
    lucky = (pd.read_csv(lucky_path, keep_default_na=False)
             if os.path.exists(lucky_path) else None)
    rng = np.random.default_rng(sel.SEED + 2210)
    rows = []
    for c in countries:
        R, m, Z = sel.sample(c, universe=U, menu_cut=False)
        Zfull = sel.zoo(c)
        Zfull.index = pd.PeriodIndex(Zfull.index, freq='M')
        Zb = pd.concat([Z, Zfull.reindex(Z.index)[[k for k in CLASSICS if k not in Z.columns]]],
                       axis=1)
        mk = m.to_numpy().ravel()
        master = ['MKT'] + list(Zb.columns)
        pos = {k: j for j, k in enumerate(master)}
        Xall = np.column_stack([mk, Zb.to_numpy()])
        cand = np.array([pos[k] for k in Z.columns])     # placebo draws come from the candidates
        zj = {k: j for j, k in enumerate(Z.columns)}
        Zv = Z.to_numpy()
        T = len(R)
        starts = rolling.window_starts(T)
        tab = sets_tab[sets_tab['country'] == c]
        named = {meth: {int(r.window): [x for x in r.members.split('|') if x]
                        for r in g.itertuples()}
                 for meth, g in tab.groupby('method')}
        if lucky is not None and (lucky['country'] == c).any():
            g = lucky[lucky['country'] == c]
            named['Lucky factors'] = {int(r.window): [x.strip() for x in
                                                      str(r.members).replace('|', ',').split(',')
                                                      if x.strip()]
                                      for r in g.itertuples()}
        named['FF5+MOM (fixed)'] = {i: CLASSICS for i in range(len(starts))}

        for sep in SEPARATIONS:
            win, dwin = {}, {}
            for i, s0 in enumerate(starts):
                end = s0 + sel.WINDOW
                lo, hi = end + STEP * (sep - 1), end + STEP * sep
                if hi > T:
                    continue
                Xw, Xe = Xall[s0:end], Xall[lo:hi]
                win[i] = (Xw.mean(0), np.cov(Xw, rowvar=False), Xe)
                zw = Zv[s0:end]
                mu_z = zw.mean(0)
                sd_z = np.where(zw.std(0) > 1e-12, zw.std(0), 1.0)
                dwin[i] = (mk[s0:end], (zw - mu_z) / sd_z, mk[lo:hi], (Zv[lo:hi] - mu_z) / sd_z)

            def report(meth, n_win, obs, null):
                for v in VARIANTS:
                    p = (1 + np.sum(null[v] >= obs[v])) / (1 + B)
                    rows.append(dict(country=c, separation=sep, method=meth, variant=v,
                                     n_windows=n_win, T_oos=len(obs['_r'][v]), sr_oos=obs[v],
                                     sr_placebo_median=np.nanmedian(null[v]),
                                     sr_placebo_q95=np.nanquantile(null[v], 0.95),
                                     p_placebo=p, mc_se=np.sqrt(p * (1 - p) / B), B=B,
                                     null=NULL_KIND.get(meth, 'label permutation'),
                                     near_005=abs(p - 0.05) <= 2 * np.sqrt(p * (1 - p) / B)))
                    print('  %s s=%d %-18s %-8s SR %5.2f  placebo median %5.2f  p %.3f  (%d windows)'
                          % (c, sep, meth, v, obs[v], np.nanmedian(null[v]), p, n_win), flush=True)

            for meth, d in named.items():
                ks = {i: len(d.get(i) or []) for i in win}
                use = [i for i in win if ks[i] > 0]
                if len(use) < 3:
                    continue
                series = chained_named(win, lambda i: ([0] + [pos[x] for x in d[i]]) if ks[i] > 0 else None)
                obs = {v: sharpe(series[v]) for v in VARIANTS}
                obs['_r'] = series
                null = {v: np.empty(B) for v in VARIANTS}
                fixed = meth == 'FF5+MOM (fixed)'
                if not fixed:
                    missing = {x for i in use for x in d[i] if x not in zj}
                    if missing:
                        raise RuntimeError('%s %s: members outside the candidates %s'
                                           % (c, meth, missing))
                for b_ in range(B):
                    if fixed:
                        S = np.concatenate([[0], rng.choice(cand, len(CLASSICS), replace=False)])
                        draw = {i: S for i in use}
                    else:
                        perm = rng.permutation(len(cand))
                        draw = {i: np.concatenate([[0], cand[perm[[zj[x] for x in d[i]]]]])
                                for i in use}
                    s_ = chained_named(win, lambda i: draw.get(i))
                    for v in VARIANTS:
                        null[v][b_] = sharpe(s_[v])
                report(meth, len(use), obs, null)

            series = chained_dense(dwin, lambda i: pca.pcs(Z.iloc[starts[i]:starts[i] + sel.WINDOW])[0][:, :K_PC])
            obs = {v: sharpe(series[v]) for v in VARIANTS}
            obs['_r'] = series
            null = {v: np.empty(B) for v in VARIANTS}
            p_ = Zv.shape[1]
            for b_ in range(B):
                V0 = db.random_directions(p_, K_PC, rng)
                s_ = chained_dense(dwin, lambda i: V0)
                for v in VARIANTS:
                    null[v][b_] = sharpe(s_[v])
            report('PC1-PC8 (fixed)', len(dwin), obs, null)

    S = pd.DataFrame(rows).set_index(['country', 'separation', 'method', 'variant'])
    S.to_csv(config.out('S1_oos_sharpe.csv'))
    print('-> %s' % config.out('S1_oos_sharpe.csv'))


if __name__ == '__main__':
    main()
