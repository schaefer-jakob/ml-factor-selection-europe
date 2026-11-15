"""Chapter 5: structure of the candidates and the test assets (Section 5.2.2,
Table 5, Table C3 Panels B and C).

  I2  Alignment of PC1-PC8 of the candidates between windows w and w + s,
      against a constant-covariance benchmark: months resampled with
      replacement from the full sample, same windows, 200 samples. The p-value
      is the share of samples with an alignment at or below the observed one.
  I3  Pearson and Spearman correlations of the stability-selection
      frequencies (P1_stability_frequencies.csv) between markets, with
      one-sided p-values from 5,000 permutations of the candidate labels.
  I4  Exposure profiles Cov(leg, factor) of the 168 legs in each window,
      demeaned across legs. (a) Share of the profile variance of all
      candidates in the first three components. (b) Share of the profile
      variance of the set selected in w + s that lies in the span of the set
      selected in w, both measured in w + s, against 500 label permutations
      of the whole sequence of sets.

Writes I2_pc_alignment_null.csv, I3_country_corr_null.csv,
I4a_exposure_rank.csv and I4b_exposure_span.csv.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402

from eufactors import config, pca, rolling          # noqa: E402
from eufactors import selection as sel              # noqa: E402

SEPARATIONS = (1, 3, 6)
B_ALIGN = 200
B_PERM = 5000
B_SPAN = 500
K_PC = sel.CAP
SPAN_METHODS = ('LASSO', 'ElasticNet', 'ElasticNet a0.2', 'ForwardStepwise',
                'LASSO CV', 'ElasticNet CV')


def _top_pcs(X, k=K_PC):
    return pca.pcs(pd.DataFrame(X))[0][:, :k]


def _mean_alignment(Zv, starts, sep):
    tops = [_top_pcs(Zv[s0:s0 + sel.WINDOW]) for s0 in starts]
    return float(np.mean([pca.alignment(tops[i], tops[i + sep]) for i in range(len(tops) - sep)]))


def alignment_null(countries, U, rng):
    rows = []
    for c in countries:
        R, m, Z = sel.sample(c, universe=U, menu_cut=False)
        Zv = Z.to_numpy()
        T = len(Zv)
        starts = rolling.window_starts(T)
        for sep in SEPARATIONS:
            obs = _mean_alignment(Zv, starts, sep)
            null = np.array([_mean_alignment(Zv[rng.integers(0, T, T)], starts, sep)
                             for _ in range(B_ALIGN)])
            p_low = (1 + np.sum(null <= obs)) / (1 + B_ALIGN)
            rows.append(dict(country=c, separation=sep, aligned_observed=obs,
                             null_mean=null.mean(), null_q05=np.quantile(null, 0.05),
                             p_observed_below_null=p_low, B=B_ALIGN))
            print('  I2 %s s=%d  observed %.3f  constant-covariance %.3f  p %.3f'
                  % (c, sep, obs, null.mean(), p_low), flush=True)
    return pd.DataFrame(rows).set_index(['country', 'separation'])


def _rank(x):
    return pd.Series(x).rank().to_numpy()


def country_correlations(rng):
    F = pd.read_csv(config.out('P1_stability_frequencies.csv'), index_col=0)
    methods = sorted({col.split(' | ')[1] for col in F.columns})
    countries = [c for c in config.COUNTRIES
                 if any(col.startswith(c + ' | ') for col in F.columns)]
    rows = []
    for meth in methods:
        for i in range(len(countries)):
            for j in range(i + 1, len(countries)):
                a = F['%s | %s' % (countries[i], meth)].to_numpy()
                b = F['%s | %s' % (countries[j], meth)].to_numpy()
                ok = ~(np.isnan(a) | np.isnan(b))
                a, b = a[ok], b[ok]
                if a.std() == 0 or b.std() == 0:
                    continue
                pear = np.corrcoef(a, b)[0, 1]
                spear = np.corrcoef(_rank(a), _rank(b))[0, 1]
                np_, ns_ = np.empty(B_PERM), np.empty(B_PERM)
                rb = _rank(b)
                for k in range(B_PERM):
                    pm = rng.permutation(len(b))
                    np_[k] = np.corrcoef(a, b[pm])[0, 1]
                    ns_[k] = np.corrcoef(_rank(a), rb[pm])[0, 1]
                rows.append(dict(method=meth, pair='%s-%s' % (countries[i], countries[j]),
                                 n=len(a), pearson=pear,
                                 p_pearson=(1 + np.sum(np_ >= pear)) / (1 + B_PERM),
                                 spearman=spear,
                                 p_spearman=(1 + np.sum(ns_ >= spear)) / (1 + B_PERM)))
    out = pd.DataFrame(rows).set_index(['method', 'pair'])
    print(out.round(3).to_string(), flush=True)
    return out


def _profiles(Rw, Fw):
    """Exposure profiles Cov(leg, factor) in one window, demeaned across legs."""
    Rc = Rw - Rw.mean(0)
    Fc = Fw - Fw.mean(0)
    P = Rc.T @ Fc / (len(Rw) - 1)
    return P - P.mean(0)


def _span_share(P_new, P_old):
    """Share of the Frobenius variance of P_new in the column span of P_old."""
    if P_old.shape[1] == 0 or P_new.shape[1] == 0:
        return np.nan
    Q, _ = np.linalg.qr(P_old)
    fit = Q @ (Q.T @ P_new)
    return float(np.sum(fit ** 2) / np.sum(P_new ** 2))


def exposure_structure(countries, U, rng):
    sets_tab = pd.read_csv(config.out('P11_window_sets.csv'), keep_default_na=False)
    rank_rows, span_rows = [], []
    for c in countries:
        R, m, Z = sel.sample(c, universe=U, menu_cut=False)
        Rv, Zv = R.to_numpy(), Z.to_numpy()
        cols = list(Z.columns)
        zj = {k: j for j, k in enumerate(cols)}
        starts = rolling.window_starts(len(Rv))
        prof = [_profiles(Rv[s0:s0 + sel.WINDOW], Zv[s0:s0 + sel.WINDOW]) for s0 in starts]
        shares = []
        for P in prof:
            sv = np.linalg.svd(P, compute_uv=False) ** 2
            shares.append(sv[:3].sum() / sv.sum())
        rank_rows.append(dict(country=c, n_windows=len(prof), top3_share_mean=np.mean(shares),
                              top3_share_min=np.min(shares), top1_share_mean=np.mean(
                                  [(np.linalg.svd(P, compute_uv=False) ** 2)[0] /
                                   (np.linalg.svd(P, compute_uv=False) ** 2).sum() for P in prof])))
        print('  I4a %s  first three components %.3f (min %.3f)'
              % (c, np.mean(shares), np.min(shares)), flush=True)
        tab = sets_tab[sets_tab['country'] == c]
        for meth in SPAN_METHODS:
            g = tab[tab['method'] == meth]
            if len(g) == 0:
                continue
            sets = {int(r.window): [zj[x] for x in r.members.split('|') if x] for r in g.itertuples()}
            for sep in SEPARATIONS:
                pairs = [(i, i + sep) for i in range(len(starts) - sep)
                         if sets.get(i) and sets.get(i + sep)]
                if len(pairs) < 3:
                    continue

                def stat(perm=None):
                    vals = []
                    for i, j in pairs:
                        a = sets[i] if perm is None else list(perm[sets[i]])
                        b = sets[j] if perm is None else list(perm[sets[j]])
                        vals.append(_span_share(prof[j][:, b], prof[j][:, a]))
                    return float(np.nanmean(vals))
                obs = stat()
                null = np.array([stat(rng.permutation(len(cols))) for _ in range(B_SPAN)])
                span_rows.append(dict(country=c, method=meth, separation=sep, n_pairs=len(pairs),
                                      span_observed=obs, span_null_mean=null.mean(),
                                      p_placebo=(1 + np.sum(null >= obs)) / (1 + B_SPAN), B=B_SPAN))
                print('  I4b %s %-16s s=%d  span %.3f  placebo %.3f  p %.3f'
                      % (c, meth, sep, obs, null.mean(), span_rows[-1]['p_placebo']), flush=True)
    return (pd.DataFrame(rank_rows).set_index('country'),
            pd.DataFrame(span_rows).set_index(['country', 'method', 'separation']))


def main():
    countries = list(config.COUNTRIES)
    U = sel.shared_universe_cut(countries)
    rng = np.random.default_rng(sel.SEED + 2410)
    alignment_null(countries, U, rng).to_csv(config.out('I2_pc_alignment_null.csv'))
    country_correlations(rng).to_csv(config.out('I3_country_corr_null.csv'))
    a, b = exposure_structure(countries, U, rng)
    a.to_csv(config.out('I4a_exposure_rank.csv'))
    b.to_csv(config.out('I4b_exposure_span.csv'))
    print('-> %s' % config.OUTPUT)


if __name__ == '__main__':
    main()
