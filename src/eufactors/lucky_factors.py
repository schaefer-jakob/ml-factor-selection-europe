"""Lucky-factors bootstrap of Harvey and Liu (2021) (Section 5.1.2,
Appendix C, Table C2).

  * Baseline model: the market. For each candidate f_j, the augmented
    time-series regressions give intercepts a+_i; the baseline gives a_i and
    their standard errors s_i.
  * Statistics (Harvey and Liu, Section 3.2), scaled by the baseline s:
        SI_mean = mean((|a+| - |a|) / s) / mean(|a| / s)
        SI_med  = (med(|a+| / s) - med(|a| / s)) / med(|a| / s)
  * Null: f*_j = f_j - delta0_j, where delta0_j is the intercept of f_j on
    [1, baseline]. f*_j cannot reduce any intercept in sample.
  * Multiple testing: bootstrap distribution of min_j SI*_j over all remaining
    candidates, months resampled i.i.d. The best candidate enters if its
    p-value is below 5 percent.

By Frisch-Waugh-Lovell, the augmented intercepts of all candidates follow from
one baseline fit: a+_j = a - delta0_j c_j with c_j = e_j'U / e_j'e_j, where U
are the baseline residuals and e_j the candidate's residual on [1, baseline].
"""
import numpy as np

ALPHA = 0.05
N_BOOT = 1000
MAX_K = 8


def _baseline(R, Fb):
    """Intercepts, their OLS standard errors, residuals and the design."""
    T = R.shape[0]
    X = np.column_stack([np.ones(T), Fb]) if Fb.size else np.ones((T, 1))
    coef = np.linalg.lstsq(X, R, rcond=None)[0]
    U = R - X @ coef
    dof = T - X.shape[1]
    s2 = (U ** 2).sum(axis=0) / dof
    xtx_inv00 = np.linalg.inv(X.T @ X)[0, 0]
    return coef[0], np.sqrt(s2 * xtx_inv00), U, X


def augmented_intercepts(R, Fb, Z):
    """(p x N) intercepts of the baseline augmented by each candidate.

    Returns (A_plus, a, s, delta0).
    """
    a, s, U, X = _baseline(R, Fb)
    G = np.linalg.lstsq(X, Z, rcond=None)[0]
    E = Z - X @ G
    delta0 = G[0]
    ee = (E ** 2).sum(axis=0)
    ee = np.where(ee > 1e-14, ee, np.inf)
    C = (E.T @ U) / ee[:, None]
    return a[None, :] - delta0[:, None] * C, a, s, delta0


def si(A_plus, a, s, stat='median'):
    """Scaled-intercept statistic per candidate."""
    if stat == 'mean':
        return (np.abs(A_plus) / s - np.abs(a) / s).mean(axis=1) / (np.abs(a) / s).mean()
    base = np.median(np.abs(a) / s)
    return (np.median(np.abs(A_plus) / s, axis=1) - base) / base


def select(R, mkt, Z, names=None, stat='median', n_boot=N_BOOT, alpha=ALPHA,
           max_k=MAX_K, rng=None):
    """Sequential lucky-factors selection.

    R (T x N) test-asset excess returns, mkt (T,), Z (T x p) candidates.
    Returns (selected names, one dict per step). The step at which the
    procedure stops is included.
    """
    rng = rng or np.random.default_rng(20261005)
    R = np.asarray(R, float)
    Z = np.asarray(Z, float)
    T, p = Z.shape
    names = list(names) if names is not None else ['f%d' % j for j in range(p)]
    base = [np.asarray(mkt, float).ravel()]
    remaining = list(range(p))
    chosen, steps = [], []
    while len(chosen) < max_k and remaining:
        Fb = np.column_stack(base)
        Zr = Z[:, remaining]
        A_plus, a, s, delta0 = augmented_intercepts(R, Fb, Zr)
        stats = si(A_plus, a, s, stat)
        jb = int(np.argmin(stats))
        actual = float(stats[jb])
        Zstar = Zr - delta0[None, :]
        boot = np.empty(n_boot)
        for b in range(n_boot):
            ix = rng.integers(0, T, T)
            Ap, ab, sb, _ = augmented_intercepts(R[ix], Fb[ix], Zstar[ix])
            boot[b] = si(Ap, ab, sb, stat).min()
        pval = float((boot <= actual).mean())
        steps.append(dict(step=len(chosen) + 1, best=names[remaining[jb]],
                          si=actual, boot_p5=float(np.percentile(boot, 5)),
                          p_multiple=pval, entered=pval < alpha))
        if pval >= alpha:
            break
        chosen.append(names[remaining[jb]])
        base.append(Z[:, remaining[jb]])
        del remaining[jb]
    return chosen, steps
