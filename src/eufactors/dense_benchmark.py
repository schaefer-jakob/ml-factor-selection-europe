"""Random dense directions in the standardized candidate space.

PC1-PC8 are combinations of all candidates. Their placebo (Table 7) is
therefore eight random orthonormal directions in the same space, with
loadings estimated in window w and applied in w + s exactly as for the
principal components.
"""
import numpy as np

N_RANDOM = 50


def random_directions(p, k, rng):
    """k orthonormal directions in R^p, uniformly distributed (QR of a Gaussian
    matrix with the sign convention that makes the draw uniform)."""
    G = rng.normal(size=(p, k))
    Q, Rr = np.linalg.qr(G)
    return Q * np.sign(np.diag(Rr))[None, :]


def project(Z_fit, Z_apply, V):
    """Apply loadings V to Z_apply, standardized with Z_fit's moments, as
    rolling.pc_factors does for the principal components."""
    X = Z_fit.to_numpy()
    mu = X.mean(0)
    sd = np.where(X.std(0) > 1e-12, X.std(0), 1.0)
    return ((Z_apply.to_numpy() - mu) / sd) @ V
