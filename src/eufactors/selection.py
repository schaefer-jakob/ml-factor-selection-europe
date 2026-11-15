"""Factor selection from the JKP candidate set (Chapter 5).

Candidates (Section 5.1.1): the JKP factors with complete returns since 1994
in every market of the scope, excluding the 56 test-asset characteristics.
That leaves 65. The definitional menu removes the four twins of a test-asset
characteristic listed in config/excluded_candidates.csv (Table A3, Panel B),
which leaves 61.

Selection methods (Section 5.1.2, equation C1):
  * LASSO and elastic net on the cross-section of average returns, with the
    intercept and the market beta as unpenalized controls. Main variant: the
    first eight candidates to enter the penalty path. Cross-validated variant:
    five-fold cross-validation over the test assets, uncapped.
  * Forward stepwise on the Fama-MacBeth t-statistic, at most eight factors.
  * The lucky-factors bootstrap is in lucky_factors.py.

Stability selection (Section 5.1.3): Meinshausen and Buehlmann (2010) on 100
random halves of the months.

The penalized regressions are solved by coordinate descent, so the code has
no dependency beyond numpy and pandas.
"""
from math import comb

import numpy as np
import pandas as pd

from . import baseline, config, data

CAP = 8                 # number of factors per selection (Appendix C)
WINDOW = 72             # rolling window in months (Section 5.1)
T_CRIT = 1.96           # forward-stepwise entry threshold
SEED = 20260909

if config.MENU not in ('definitional', 'full'):
    raise ValueError('candidate menu must be definitional or full, got %r' % config.MENU)
if config.CANDIDATE_BETAS not in ('joint', 'bivariate'):
    raise ValueError('candidate betas must be joint or bivariate, got %r'
                     % config.CANDIDATE_BETAS)


# ------------------------------------------------------------------- data ---
def zoo(country):
    """All JKP factors except the 56 test-asset characteristics."""
    df = data.factors_raw(country)
    df['date'] = pd.to_datetime(df['date'])
    w = df.pivot(index='date', columns='name', values='ret')
    return w[[c for c in w.columns if c not in baseline.CHARS]]


def shared_universe(countries=('DEU', 'GBR'), start='1994-01-01'):
    """Factors with complete returns since `start` in every market in scope,
    so that every market selects from the same candidates."""
    sets = []
    for c in countries:
        Z = zoo(c).loc[start:]
        Z.index = pd.PeriodIndex(Z.index, freq='M')
        sets.append(set(Z.columns[Z.notna().all()]))
    return sorted(set.intersection(*sets))


def excluded_candidates():
    """The four twins of Table A3, Panel B."""
    return pd.read_csv(config.config('excluded_candidates.csv'))['candidate'].tolist()


def apply_menu(Z, menu=None):
    """Drop the excluded twins under the definitional menu; keep all under
    the full menu (Table C6)."""
    menu = (menu or config.MENU).lower()
    if menu == 'full':
        return Z
    ex = set(excluded_candidates())
    return Z[[c for c in Z.columns if c not in ex]]


def shared_universe_cut(countries=('DEU', 'GBR'), start='1994-01-01'):
    """The candidate set: shared universe after the menu rule."""
    keep = None
    for c in countries:
        _, _, Z = sample(c, start=start, universe=shared_universe(countries, start))
        keep = set(Z.columns) if keep is None else (keep & set(Z.columns))
    return sorted(keep)


def sample(country, start='1994-01-01', weighting='vw_cap', universe=None,
           menu_cut=True):
    """Test-asset returns R, market factor and candidates Z on the balanced
    panel of months in which every leg has a return (Section 5.1)."""
    legs = baseline.leg_panel(country, start=start, weighting=weighting)
    R = legs.pivot_table(index='datadate', columns='gvkey',
                         values='ret_usd_excess')
    Z = zoo(country)
    mkt = data.classic_factors(country.lower())[['MKT']]
    R.index = pd.PeriodIndex(R.index, freq='M')
    Z.index = pd.PeriodIndex(Z.index, freq='M')
    mkt.index = pd.PeriodIndex(mkt.index, freq='M')
    if universe is not None:
        Z = Z[list(universe)]
    idx = R.dropna(axis=1, how='all').dropna().index
    idx = idx.intersection(mkt.dropna().index)
    Z = Z.loc[Z.index.isin(idx)].dropna(axis=1)
    idx = idx.intersection(Z.index)
    Z = Z.loc[idx]
    if menu_cut:
        Z = apply_menu(Z)
    return (R.loc[idx].dropna(axis=1), mkt.loc[idx], Z)


# ----------------------------------------------------- Fama-MacBeth kernel ---
def fmb(R, F):
    """Two-pass Fama-MacBeth with betas held constant within the sample.

    R is (T x N), F is (T x K). Returns (lambda_mean, t_stats), both of length
    K + 1 with gamma0 first. The t-statistics use the plain Fama-MacBeth
    standard error (Section 5.1.2).
    """
    T = R.shape[0]
    X = np.column_stack([np.ones(T), F])
    B = np.linalg.lstsq(X, R, rcond=None)[0][1:]
    Z = np.column_stack([np.ones(B.shape[1]), B.T])
    lam = np.linalg.lstsq(Z, R.T, rcond=None)[0].T
    m = lam.mean(axis=0)
    se = lam.std(axis=0, ddof=1) / np.sqrt(T)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(se > 0, m / se, 0.0)
    return m, t


def betas(R, F):
    """(N x K) betas from one time-series regression on all columns of F."""
    X = np.column_stack([np.ones(R.shape[0]), F])
    return np.linalg.lstsq(X, R, rcond=None)[0][1:].T


def bivariate_betas(R, Fmkt, Z):
    """(N x K) betas, each from a regression on the market and one candidate
    only (Appendix C check, and the exposure rank of Table C3)."""
    Rv = R.to_numpy() if hasattr(R, 'to_numpy') else R
    Zv = Z.to_numpy() if hasattr(Z, 'to_numpy') else Z
    T = Rv.shape[0]
    ones = np.ones(T)
    out = np.empty((Rv.shape[1], Zv.shape[1]))
    for j in range(Zv.shape[1]):
        X = np.column_stack([ones, Fmkt, Zv[:, j]])
        out[:, j] = np.linalg.lstsq(X, Rv, rcond=None)[0][2]
    return out


# ------------------------------------------------------ forward stepwise ---
def forward_stepwise(R, Fmkt, Z, cap=CAP, t_crit=T_CRIT):
    """Forward stepwise selection (Section 5.1.2).

    Starting from the market, add the candidate whose premium has the highest
    absolute Fama-MacBeth t-statistic, until no candidate reaches t_crit or
    `cap` candidates have entered. Returns (selected names, cap reached).
    """
    names = list(Z.columns)
    Zv = Z.to_numpy()
    chosen, chosen_ix = [], []
    while len(chosen) < cap:
        best_t, best = -np.inf, None
        for j in range(Zv.shape[1]):
            if j in chosen_ix:
                continue
            F = np.column_stack([Fmkt] + [Zv[:, k] for k in chosen_ix] + [Zv[:, j]])
            _, t = fmb(R, F)
            tj = abs(t[-1])
            if np.isfinite(tj) and tj > best_t:
                best_t, best = tj, j
        if best is None or best_t < t_crit:
            return chosen, False
        chosen_ix.append(best)
        chosen.append(names[best])
    return chosen, True


# ------------------------------------------- coordinate descent (L1 / L1+L2) ---
def _soft(z, g):
    return np.sign(z) * np.maximum(np.abs(z) - g, 0.0)


def coord_descent(X, y, lam, alpha=1.0, n_iter=1000, tol=1e-8, b0=None):
    """Minimize 1/(2n) ||y - Xb||^2 + lam (alpha |b|_1 + (1 - alpha)/2 ||b||^2)
    (equation C1). alpha = 1 is the LASSO. X standardized, y centred, no
    intercept."""
    n, p = X.shape
    b = np.zeros(p) if b0 is None else b0.copy()
    r = y - X @ b
    norms = (X ** 2).sum(axis=0) / n
    for _ in range(n_iter):
        delta = 0.0
        for j in range(p):
            if b[j] != 0.0:
                r = r + X[:, j] * b[j]
            rho = X[:, j] @ r / n
            new = _soft(rho, lam * alpha) / (norms[j] + lam * (1 - alpha))
            if new != 0.0:
                r = r - X[:, j] * new
            delta = max(delta, abs(new - b[j]))
            b[j] = new
        if delta < tol:
            break
    return b


def _prep(B, rbar):
    """Standardize the candidate betas and the average returns. Standardizing
    y puts the penalty on the scale of the residual, so that the L2 term of
    the elastic net is effective."""
    mu, sd = B.mean(axis=0), B.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    ys = rbar.std(ddof=1)
    return (B - mu) / sd, (rbar - rbar.mean()) / (ys if ys > 1e-12 else 1.0)


def _lam_max(X, y, alpha=1.0):
    """Smallest penalty at which every coefficient is zero."""
    return np.abs(X.T @ y).max() / (len(y) * max(alpha, 1e-8))


def first_q_path(X, y, alpha, q, n_lam=60, ratio=1e-3):
    """Lower the penalty from lambda_max and return the first q variables to
    enter (the first-eight rule of Section 5.1.2)."""
    lmax = _lam_max(X, y, alpha)
    b = np.zeros(X.shape[1])
    for L in np.geomspace(lmax, lmax * ratio, n_lam):
        b = coord_descent(X, y, L, alpha, n_iter=300, b0=b)
        nz = np.flatnonzero(b)
        if len(nz) >= q:
            return list(nz[np.argsort(-np.abs(b[nz]))][:q])
    nz = np.flatnonzero(b)
    return list(nz[np.argsort(-np.abs(b[nz]))][:q])


def candidate_betas(Rv, Fmkt, Z):
    """Candidate betas of the penalized cross-section: joint (default) or
    bivariate (Appendix C check), as set by the scope."""
    if config.CANDIDATE_BETAS == 'bivariate':
        return bivariate_betas(Rv, Fmkt, Z)
    T, p_cand = Rv.shape[0], Z.shape[1]
    if not T > p_cand + 3:
        raise ValueError('cannot identify %d candidate betas from %d months'
                         % (p_cand, T))
    return betas(Rv, np.column_stack([Fmkt, Z.to_numpy()]))[:, 1:]


def penalised_design(R, Fmkt, Z):
    """Standardized (X, y) of equation (C1): candidate betas and average
    returns, both after regressing them across test assets on an intercept
    and the market beta."""
    Rv = R.to_numpy() if hasattr(R, 'to_numpy') else R
    Bz = candidate_betas(Rv, Fmkt, Z)
    Bm = betas(Rv, Fmkt.reshape(-1, 1))
    rbar = Rv.mean(axis=0)
    Xm = np.column_stack([np.ones(len(rbar)), Bm])
    y = rbar - Xm @ np.linalg.lstsq(Xm, rbar, rcond=None)[0]
    Bz = Bz - Xm @ np.linalg.lstsq(Xm, Bz, rcond=None)[0]
    return _prep(Bz, y)


def penalised_select(R, Fmkt, Z, alpha=1.0, cap=CAP, rng=None):
    """LASSO (alpha = 1) or elastic net (alpha < 1) with the first-q rule.
    Returns (selected names, None)."""
    X, y = penalised_design(R, Fmkt, Z)
    keep = first_q_path(X, y, alpha, cap)
    return [Z.columns[i] for i in keep], None


def _asset_groups(R, n_folds=5):
    """Cross-validation fold of each test asset. The three legs of one
    characteristic always share a fold (Appendix C); in the pooled market the
    Swiss and Swedish legs of a characteristic share it as well."""
    def _char(c):
        return c.rsplit('_pf', 1)[0].split('.')[-1]
    cols = list(R.columns) if hasattr(R, 'columns') else [str(i) for i in range(R.shape[1])]
    chars = sorted({_char(c) for c in cols})
    order = np.random.default_rng(SEED).permutation(len(chars))
    fold_of = {chars[j]: i % n_folds for i, j in enumerate(order)}
    return np.array([fold_of[_char(c)] for c in cols])


def penalised_select_cv(R, Fmkt, Z, alpha=1.0, n_folds=5, n_lam=60, ratio=1e-3):
    """LASSO or elastic net with the penalty chosen by five-fold
    cross-validation over test assets, without a cap (LASSO CV)."""
    X, y = penalised_design(R, Fmkt, Z)
    folds = _asset_groups(R, n_folds)
    lmax = _lam_max(X, y, alpha)
    lams = np.geomspace(lmax, lmax * ratio, n_lam)
    err = np.zeros(n_lam)
    for k in range(n_folds):
        tr, te = folds != k, folds == k
        mx, my = X[tr].mean(axis=0), y[tr].mean()
        Xtr, ytr = X[tr] - mx, y[tr] - my
        b = np.zeros(X.shape[1])
        for i, L in enumerate(lams):
            b = coord_descent(Xtr, ytr, L, alpha, n_iter=300, b0=b)
            err[i] += ((y[te] - my - (X[te] - mx) @ b) ** 2).sum()
    L_best = lams[int(np.argmin(err))]
    b = np.zeros(X.shape[1])
    for L in lams[lams >= L_best]:
        b = coord_descent(X, y, L, alpha, n_iter=300, b0=b)
    return [Z.columns[i] for i in np.flatnonzero(b)], None


def factor_clusters(Z, min_avg_corr=0.5):
    """Average-linkage clusters of candidates on |correlation|, merged while
    the average absolute correlation between two clusters is at least
    min_avg_corr (Table C3, Panel A). Returns {factor: cluster id}."""
    C = np.abs(np.corrcoef(Z.to_numpy().T))
    clusters = [[i] for i in range(C.shape[0])]
    while len(clusters) > 1:
        best, pair = -1.0, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                s = C[np.ix_(clusters[a], clusters[b])].mean()
                if s > best:
                    best, pair = s, (a, b)
        if best < min_avg_corr:
            break
        a, b = pair
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    return {Z.columns[i]: k for k, cl in enumerate(clusters) for i in cl}


# ------------------------------------------------------ stability selection ---
def stability_selection(selector, R, Fmkt, Z, n_draws=100, frac=0.5, rng=None,
                        return_empty=False):
    """Selection frequency of each candidate over n_draws random halves of the
    months (Meinshausen and Buehlmann, 2010). Returns (frequencies, share of
    draws in which the cap binds, NaN where not applicable), and with
    return_empty also the share of draws in which nothing is selected."""
    rng = rng or np.random.default_rng(SEED)
    Fmkt = np.asarray(Fmkt).ravel()
    T = R.shape[0]
    k = int(round(frac * T))
    counts, caps, empty = {}, 0, 0
    for _ in range(n_draws):
        take = np.sort(rng.choice(T, size=k, replace=False))
        sel, capped = selector(R.iloc[take], Fmkt[take], Z.iloc[take])
        if capped is None:
            caps = None
        elif caps is not None:
            caps += int(capped)
        empty += int(len(sel) == 0)
        for f in sel:
            counts[f] = counts.get(f, 0) + 1
    freq = pd.Series(counts, dtype=float).sort_values(ascending=False) / n_draws
    cap_share = np.nan if caps is None else caps / n_draws
    if return_empty:
        return freq, cap_share, empty / n_draws
    return freq, cap_share


def null_frequencies(selector, R, Fmkt, Z, n_draws=100, frac=0.5, rng=None):
    """Highest selection frequency when no candidate carries a premium
    (Table C1, Panel C). The candidate returns are kept; the test-asset
    returns are replaced by market exposure plus noise."""
    rng = rng or np.random.default_rng(SEED)
    Fmkt = np.asarray(Fmkt).ravel()
    Rv = R.to_numpy() if hasattr(R, 'to_numpy') else R
    bm = rng.normal(1.0, 0.15, Rv.shape[1])
    sd = float(np.std(Rv - np.outer(Fmkt, bm)))
    Rnull = pd.DataFrame(np.outer(Fmkt, bm) + rng.normal(0, sd, Rv.shape),
                         index=R.index)
    freq, _ = stability_selection(selector, Rnull, Fmkt, Z,
                                  n_draws=n_draws, frac=frac, rng=rng)
    return float(freq.max()) if len(freq) else 0.0


def mb_bound(q, pi, p):
    """Meinshausen-Buehlmann bound on the expected number of false selections,
    q^2 / ((2 pi - 1) p) (Appendix C)."""
    return q * q / ((2 * pi - 1) * p)


# ------------------------------------------------------------- overlap ---
def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else np.nan


def jaccard_random(k1, k2, p):
    """Expected Jaccard index of two random subsets of sizes k1 and k2 drawn
    from p candidates, summed over the hypergeometric distribution of the
    intersection. Zero if exactly one set is empty, NaN if both are."""
    if k1 == 0 and k2 == 0:
        return np.nan
    if k1 == 0 or k2 == 0:
        return 0.0
    lo, hi = max(0, k1 + k2 - p), min(k1, k2)
    tot = comb(p, k2)
    return float(sum(comb(k1, i) * comb(p - k1, k2 - i) / tot * i / (k1 + k2 - i)
                     for i in range(lo, hi + 1)))


# ----------------------------------------------------------------- methods ---
def selectors(cap=CAP):
    """Forward stepwise, LASSO and elastic net (alpha = 0.5) with cap q."""
    return {
        'ForwardStepwise': lambda R, m, Z: forward_stepwise(R, m, Z, cap=cap),
        'LASSO': lambda R, m, Z: penalised_select(R, m, Z, alpha=1.0, cap=cap),
        'ElasticNet': lambda R, m, Z: penalised_select(R, m, Z, alpha=0.5, cap=cap),
    }


def selectors_extended(cap=CAP):
    """selectors() plus the elastic net with alpha = 0.2 and the
    cross-validated LASSO and elastic net."""
    s = selectors(cap)
    s['ElasticNet a0.2'] = lambda R, m, Z: penalised_select(R, m, Z, alpha=0.2, cap=cap)
    s['LASSO CV'] = lambda R, m, Z: penalised_select_cv(R, m, Z, alpha=1.0)
    s['ElasticNet CV'] = lambda R, m, Z: penalised_select_cv(R, m, Z, alpha=0.5)
    return s
