"""Principal components of the candidates and subspace alignment (Section
5.2.2, Table C3, Panel B).

Components rotate between windows, so their identity cannot be compared
directly. The alignment of two subspaces is measured through the principal
angles between them, which does not depend on the basis.
"""
import numpy as np


def pcs(Zw):
    """Principal components of the correlation matrix of one window.

    Returns (V, S, share): orthonormal loadings V (p x p) ordered by explained
    variance, component series S (T x p) and the variance shares.
    """
    X = Zw.to_numpy()
    Xs = (X - X.mean(0)) / np.where(X.std(0) > 1e-12, X.std(0), 1.0)
    C = np.corrcoef(Xs.T)
    w, V = np.linalg.eigh(C)
    o = np.argsort(w)[::-1]
    w, V = np.clip(w[o], 0, None), V[:, o]
    return V, Xs @ V, w / w.sum()


def alignment(A, B):
    """||A'B||_F^2 / min(k1, k2) for orthonormal A (p x k1) and B (p x k2),
    the mean squared cosine of the principal angles: 1 if one subspace
    contains the other, 0 if they are orthogonal."""
    if A.shape[1] == 0 or B.shape[1] == 0:
        return np.nan
    return float(np.sum((A.T @ B) ** 2) / min(A.shape[1], B.shape[1]))


def subspace_null(tops_i, tops_j, k1, k2, rng, n=200):
    """Mean alignment when k1 and k2 columns are drawn at random from the two
    windows' leading component bases."""
    if k1 == 0 or k2 == 0:
        return np.nan
    vals = [alignment(tops_i[:, rng.choice(tops_i.shape[1], k1, replace=False)],
                      tops_j[:, rng.choice(tops_j.shape[1], k2, replace=False)])
            for _ in range(n)]
    return float(np.nanmean(vals))
