"""SDF shrinkage of Kozak, Nagel and Santosh (2020), "Shrinking the
Cross-Section" (Section 5.1.4, Table C5).

  * L2 only, their equation (22):  b = (Sigma + gamma I)^-1 mu,
    gamma = tau / (kappa^2 T), tau = tr(Sigma). kappa is the prior root
    expected maximum squared Sharpe ratio.
  * L1 and L2, their equation (28):
        min (mu - Sigma b)' Sigma^-1 (mu - Sigma b) + gamma2 b'b + gamma1 |b|_1,
    solved by coordinate descent on the expanded objective
    b' Sigma b - 2 mu' b + gamma2 b'b + gamma1 |b|_1.
  * Penalty choice: three-fold cross-validation over contiguous blocks of
    months, scored by the out-of-sample R^2 of their equation (30),
    1 - (mu2 - Sigma2 b)'(mu2 - Sigma2 b) / mu2' mu2.
  * Data: candidates orthogonalized to the market with betas from the
    estimation sample and rescaled to the market's standard deviation.
"""
import numpy as np

KAPPAS = np.geomspace(0.01, 10.0, 25)
SPARSITY = (1, 2, 3, 4, 5, 6, 7, 8)
N_FOLDS = 3


# ----------------------------------------------------------- data treatment ---
def prepare(Z, m, beta=None, scale=None):
    """Orthogonalize the candidates to the market and rescale them.

    beta and scale from the estimation window are passed when transforming an
    evaluation window, so no information from the evaluation window is used.
    """
    Z = np.asarray(Z, float)
    m = np.asarray(m, float).ravel()
    if beta is None:
        X = np.column_stack([np.ones(len(m)), m])
        beta = np.linalg.lstsq(X, Z, rcond=None)[0][1]
    F = Z - np.outer(m, beta)
    if scale is None:
        sd = F.std(axis=0, ddof=1)
        scale = m.std(ddof=1) / np.where(sd > 1e-12, sd, 1.0)
    return F * scale, beta, scale


def moments(F):
    F = np.asarray(F, float)
    return F.mean(axis=0), np.atleast_2d(np.cov(F, rowvar=False, ddof=1))


def gamma_from_kappa(kappa, Sigma, T):
    """gamma = tr(Sigma) / (kappa^2 T)."""
    return np.trace(Sigma) / (kappa ** 2 * T)


# --------------------------------------------------------------- estimators ---
def l2_sdf(mu, Sigma, gamma):
    """Dense SDF coefficients, KNS equation (22)."""
    return np.linalg.solve(Sigma + gamma * np.eye(len(mu)), mu)


def dual_penalty(mu, Sigma, gamma2, gamma1, b0=None, n_iter=5000, tol=1e-11):
    """KNS equation (28) by cyclic coordinate descent.

    Coordinate update:
        b_j = soft(mu_j - sum_{k != j} Sigma_jk b_k, gamma1 / 2) / (Sigma_jj + gamma2)
    With gamma1 = 0 this reproduces l2_sdf.
    """
    p = len(mu)
    b = np.zeros(p) if b0 is None else np.array(b0, float)
    Sb = Sigma @ b
    diag = np.diag(Sigma) + gamma2
    h = gamma1 / 2.0
    for _ in range(n_iter):
        delta = 0.0
        for j in range(p):
            r = mu[j] - Sb[j] + Sigma[j, j] * b[j]
            new = np.sign(r) * max(abs(r) - h, 0.0) / diag[j]
            d = new - b[j]
            if d != 0.0:
                Sb += Sigma[:, j] * d
                b[j] = new
                delta = max(delta, abs(d))
        if delta < tol:
            break
    return b


def sparsity_path(mu, Sigma, gamma2, ks=SPARSITY, n_g1=80, ratio=1e-5):
    """{k: (b, number of non-zeros)} for each target number of non-zero
    coefficients k.

    gamma1 is lowered from 2 max|mu|, the value at which b = 0. Where the path
    jumps past k, gamma1 is bisected; if k is still not hit, the solution with
    more than k non-zeros is kept and its true count returned.
    """
    g1max = 2.0 * np.abs(mu).max()
    out, b = {}, np.zeros(len(mu))
    g_prev, b_prev = g1max, b.copy()
    for g1 in np.geomspace(g1max, g1max * ratio, n_g1):
        b = dual_penalty(mu, Sigma, gamma2, g1, b0=b, n_iter=500, tol=1e-9)
        nz = int((b != 0).sum())
        for k in ks:
            if k in out or nz < k:
                continue
            if nz == k:
                out[k] = (b.copy(), nz)
                continue
            lo, hi, bb = g1, g_prev, b_prev.copy()
            best = (b.copy(), nz)
            for _ in range(25):
                mid = np.sqrt(lo * hi)
                bm = dual_penalty(mu, Sigma, gamma2, mid, b0=bb, n_iter=500, tol=1e-9)
                nm = int((bm != 0).sum())
                if nm == k:
                    best = (bm, nm)
                    break
                if nm > k:
                    lo, best = mid, (bm, nm)
                else:
                    hi, bb = mid, bm
            out[k] = best
        if all(k in out for k in ks):
            break
        g_prev, b_prev = g1, b.copy()
    return out


def r2_oos(mu2, Sigma2, b):
    """KNS equation (30)."""
    e = mu2 - Sigma2 @ b
    return 1.0 - (e @ e) / (mu2 @ mu2)


# --------------------------------------------------------- cross-validation ---
def _folds(T, k=N_FOLDS):
    edges = np.linspace(0, T, k + 1).astype(int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(k)]


def cv_surface(F, kappas=KAPPAS, ks=SPARSITY, rotate_pc=False):
    """Cross-validated R^2 (equation 30) on the grid kappa x number of
    non-zeros. Column 'dense' is the L2-only estimator.

    rotate_pc: rotate into the principal components of the training folds'
    covariance before estimating. The dense solution is invariant to this
    rotation; only the sparse ones change.
    """
    F = np.asarray(F, float)
    T = F.shape[0]
    cols = ['dense'] + list(ks)
    acc = np.zeros((len(kappas), len(cols)))
    folds = _folds(T)
    for te in folds:
        tr = np.setdiff1d(np.arange(T), te)
        mu1, S1 = moments(F[tr])
        mu2, S2 = moments(F[te])
        if rotate_pc:
            Q = np.linalg.eigh(S1)[1][:, ::-1]
            mu1, S1, mu2, S2 = Q.T @ mu1, Q.T @ S1 @ Q, Q.T @ mu2, Q.T @ S2 @ Q
        for i, kap in enumerate(kappas):
            g2 = gamma_from_kappa(kap, S1, len(tr))
            acc[i, 0] += r2_oos(mu2, S2, l2_sdf(mu1, S1, g2))
            if not ks:
                continue
            path = sparsity_path(mu1, S1, g2, ks)
            for j, k in enumerate(ks, start=1):
                acc[i, j] += r2_oos(mu2, S2, path[k][0]) if k in path else np.nan
    return acc / len(folds), cols


def fit_cv(F, kappas=KAPPAS, ks=SPARSITY, rotate_pc=False):
    """Choose kappa by cross-validation for each column and refit on all of F.

    Returns ({col: (b in the original factor space, kappa, non-zeros)},
    surface, columns).
    """
    surf, cols = cv_surface(F, kappas, ks, rotate_pc)
    mu, S = moments(F)
    Q = np.linalg.eigh(S)[1][:, ::-1] if rotate_pc else np.eye(len(mu))
    muq, Sq = Q.T @ mu, Q.T @ S @ Q
    out = {}
    for j, col in enumerate(cols):
        sc = surf[:, j]
        if np.all(np.isnan(sc)):
            continue
        kap = kappas[int(np.nanargmax(sc))]
        g2 = gamma_from_kappa(kap, Sq, F.shape[0])
        if col == 'dense':
            bq, nz = l2_sdf(muq, Sq, g2), len(mu)
        else:
            path = sparsity_path(muq, Sq, g2, (col,))
            if col not in path:
                continue
            bq, nz = path[col]
        out[col] = (Q @ bq, kap, nz)
    return out, surf, cols
