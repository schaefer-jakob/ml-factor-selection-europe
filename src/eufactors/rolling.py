"""Rolling-window selection and out-of-sample evaluation (Sections 5.1.3 and
5.1.4).

Each method selects factors in 26 windows of 72 months, moved forward by 12
months. Overlap between the sets of windows w and w + s is measured by the
Jaccard index relative to random sets of the same sizes. Out of sample, only
the identity of the factors selected in w is carried to w + s; betas and
premia are re-estimated there and the cross-sectional R^2 is computed.
"""
import numpy as np
import pandas as pd

from . import data, fmb, pca
from . import selection as sel

STEP = 12                                  # months between window starts
SEPARATIONS_OOS = (1, 3, 6)                # s = 6 shares no month with w
SEPARATIONS_OVERLAP = (1, 2, 3, 5, 6, 10)
N_RANDOM = 50                              # random sets per window pair
CLASSICS = list(data.CLASSICS)             # characteristics of the six-factor model
FIXED_MODELS = ['FF5+MOM (fixed)', 'PC1-PC8 (fixed)']


def window_starts(T):
    return list(range(0, T - sel.WINDOW + 1, STEP))


def csr2_F(Rw, Fw):
    """Cross-sectional OLS R^2 of average returns on full-window betas."""
    Rv = Rw.to_numpy() if hasattr(Rw, 'to_numpy') else Rw
    B = sel.betas(Rv, Fw)
    rbar = Rv.mean(axis=0)
    X = np.column_stack([np.ones(len(rbar)), B])
    fit = X @ np.linalg.lstsq(X, rbar, rcond=None)[0]
    return 1 - ((rbar - fit) ** 2).sum() / ((rbar - rbar.mean()) ** 2).sum()


def csr2(Rw, mkw, Zw, cols):
    """csr2_F for the market plus the named candidates `cols`."""
    return csr2_F(Rw, np.column_stack([mkw, Zw[list(cols)].to_numpy()]))


def pc_factors(Z_fit, Z_apply, k=sel.CAP):
    """Returns of PC1..PCk with loadings estimated on Z_fit and applied to
    Z_apply, standardized with Z_fit's moments."""
    X = Z_fit.to_numpy()
    mu, sd = X.mean(0), np.where(X.std(0) > 1e-12, X.std(0), 1.0)
    V, _, _ = pca.pcs(Z_fit)
    return ((Z_apply.to_numpy() - mu) / sd) @ V[:, :k]


def select_in_windows(samples, methods, verbose=True):
    """Run every method in every rolling window.

    samples: {country: (R, market, Z)}; methods: {name: selector}.
    Returns (sets, capped, starts) with sets[(country, method)] a list of
    selected sets, one per window.
    """
    sets, capped, starts = {}, {}, {}
    for c, (R, m, Z) in samples.items():
        mk = m.to_numpy().ravel()
        starts[c] = window_starts(len(R))
        for meth, fn in methods.items():
            per, cps = [], []
            for s0 in starts[c]:
                sl = slice(s0, s0 + sel.WINDOW)
                pick, cp = fn(R.iloc[sl], mk[sl], Z.iloc[sl])
                per.append(set(pick))
                cps.append(cp)
            sets[(c, meth)] = per
            capped[(c, meth)] = cps
            if verbose:
                print('  %s %-16s windows done, mean K %.1f'
                      % (c, meth, np.mean([len(x) for x in per])), flush=True)
    return sets, capped, starts


def window_sets_table(sets, starts):
    """The selected set of every method and window (P11_window_sets.csv)."""
    return pd.DataFrame(
        [dict(country=c, method=meth, window=i, start=starts[c][i], K=len(s_),
              members='|'.join(sorted(s_)))
         for (c, meth), per in sets.items() for i, s_ in enumerate(per)]
    ).set_index(['country', 'method', 'window'])


def overlap_by_separation(sets, p):
    """Mean Jaccard index of the sets of windows w and w + s, and its ratio to
    random sets of the same sizes (Figure 1)."""
    rows = []
    for (c, meth), per in sets.items():
        for lag in SEPARATIONS_OVERLAP:
            prs = [(per[i], per[i + lag]) for i in range(len(per) - lag)]
            if not prs:
                continue
            valid = [(a, b) for a, b in prs if (a or b)]
            obs = np.nanmean([sel.jaccard(a, b) for a, b in prs])
            nul = np.nanmean([sel.jaccard_random(len(a), len(b), p) for a, b in prs])
            rows.append(dict(country=c, method=meth, separation=lag,
                             months_shared=max(0, sel.WINDOW - lag * STEP),
                             n_pairs=len(valid),
                             observed=obs, random=nul,
                             ratio=obs / nul if nul else np.nan))
    return pd.DataFrame(rows).set_index(['country', 'method', 'separation'])


def cluster_overlap(samples, sets, methods):
    """Overlap of the selected clusters of near-substitute candidates
    (average |corr| >= 0.5) relative to random sets of the same sizes
    (Table C3, Panel A). The null is simulated with 100 draws per pair."""
    rows = []
    for c, (_, _, Z) in samples.items():
        cmap = sel.factor_clusters(Z)
        n_cl = len(set(cmap.values()))
        names = np.array(list(Z.columns))
        rng = np.random.default_rng(sel.SEED + 7)
        for meth in methods:
            per = [{cmap[f_] for f_ in s_} for s_ in sets[(c, meth)]]
            raw = sets[(c, meth)]
            for lag in SEPARATIONS_OVERLAP:
                idx = [(i, i + lag) for i in range(len(per) - lag)]
                if not idx:
                    continue
                obs = np.nanmean([sel.jaccard(per[i], per[j]) for i, j in idx])
                sims = []
                for i, j in idx:
                    ki, kj = len(raw[i]), len(raw[j])
                    if ki == 0 and kj == 0:
                        continue
                    v = []
                    for _ in range(100):
                        a = {cmap[x] for x in rng.choice(names, ki, replace=False)}
                        b = {cmap[x] for x in rng.choice(names, kj, replace=False)}
                        v.append(sel.jaccard(a, b) if (a or b) else np.nan)
                    sims.append(np.nanmean(v))
                nul = np.nanmean(sims) if sims else np.nan
                rows.append(dict(country=c, method=meth, separation=lag,
                                 n_clusters=n_cl, observed=obs, random=nul,
                                 ratio=obs / nul if nul else np.nan))
    return pd.DataFrame(rows).set_index(['country', 'method', 'separation'])


def out_of_sample_r2(samples, sets, starts, methods, Z_bench=None, verbose=True):
    """Out-of-sample cross-sectional R^2 at the separations SEPARATIONS_OOS
    (Table 6, R^2 columns).

    For every window pair (w, w + s) the factors selected in w are evaluated
    in w + s, against the mean of N_RANDOM random candidate sets of the same
    size. methods may include FIXED_MODELS: the six-factor model (taken from
    Z_bench) and PC1-PC8 with loadings estimated in w. Windows in which a
    method selects nothing are skipped. Returns (summary, pairs).
    """
    rng = np.random.default_rng(sel.SEED)
    rows, pair_rows = [], []
    for c, (R, m, Z) in samples.items():
        mk = m.to_numpy().ravel()
        st = starts[c]
        for sep in SEPARATIONS_OOS:
            for meth in methods:
                ins, oos, roos, ks = [], [], [], []
                for i in range(len(st) - sep):
                    a_ = slice(st[i], st[i] + sel.WINDOW)
                    b_ = slice(st[i + sep], st[i + sep] + sel.WINDOW)
                    if meth == 'FF5+MOM (fixed)':
                        pick = list(CLASSICS)
                    elif meth == 'PC1-PC8 (fixed)':
                        pick = None
                    else:
                        pick = sorted(sets[(c, meth)][i])
                        if not pick:
                            continue
                    if pick is None:
                        k = sel.CAP
                        Fa = np.column_stack([mk[a_], pc_factors(Z.iloc[a_], Z.iloc[a_])])
                        Fb = np.column_stack([mk[b_], pc_factors(Z.iloc[a_], Z.iloc[b_])])
                        r_in, r_out = csr2_F(R.iloc[a_], Fa), csr2_F(R.iloc[b_], Fb)
                    else:
                        k = len(pick)
                        Zp = Z_bench[c] if meth == 'FF5+MOM (fixed)' else Z
                        r_in = csr2(R.iloc[a_], mk[a_], Zp.iloc[a_], pick)
                        r_out = csr2(R.iloc[b_], mk[b_], Zp.iloc[b_], pick)
                    rr = [csr2(R.iloc[b_], mk[b_], Z.iloc[b_],
                               rng.choice(list(Z.columns), k, replace=False))
                          for _ in range(N_RANDOM)]
                    ins.append(r_in)
                    oos.append(r_out)
                    roos.append(float(np.mean(rr)))
                    ks.append(k)
                    pair_rows.append(dict(country=c, method=meth, separation=sep,
                                          window=i, K=k, R2_in=r_in, R2_out=r_out,
                                          R2_out_random=float(np.mean(rr))))
                if len(oos) < 3:
                    continue
                d = np.array(oos) - np.array(roos)
                se_iid = d.std(ddof=1) / np.sqrt(len(d))
                se_nw = fmb.newey_west_se(d)
                rows.append(dict(country=c, method=meth, separation=sep,
                                 months_shared=max(0, sel.WINDOW - sep * STEP),
                                 n_pairs=len(d), mean_K=np.mean(ks),
                                 R2_in_selected=np.mean(ins),
                                 R2_out_selected=np.mean(oos),
                                 R2_out_random=np.mean(roos),
                                 edge_out=d.mean(),
                                 t_edge_iid=d.mean() / se_iid,
                                 t_edge_nw=d.mean() / se_nw,
                                 windows_won=int((d > 0).sum())))
                if verbose:
                    print('  %s s=%d %-18s edge %+.3f (t_NW=%.2f, %d pairs)'
                          % (c, sep, meth, d.mean(), d.mean() / se_nw, len(d)), flush=True)
    return (pd.DataFrame(rows).set_index(['country', 'separation', 'method']),
            pd.DataFrame(pair_rows).set_index(['country', 'separation', 'method', 'window']))
