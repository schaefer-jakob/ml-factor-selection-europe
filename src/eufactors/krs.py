"""Misspecification-robust standard errors of Kan, Robotti and Shanken (2013).

Used for the KRS rows of Table 3 and the KRS columns of Tables B4 and B5:
full-sample betas, two-pass OLS cross-sectional regression, standard errors
that remain valid when the model is misspecified (Section 4.1.2).

With mu_R and mu_f the sample means, Sigma_f the factor covariance (divided
by T), B = Cov(R, f) Sigma_f^-1, X = [1, B], H = (X'X)^-1 and
gamma = H X' mu_R, the delta method gives the influence series

    e_t   = R_t - mu_R - B (f_t - mu_f)        time-series residuals
    z_t   = Sigma_f^-1 (f_t - mu_f)
    e     = mu_R - X gamma                      pricing errors
    u_t   = e' e_t
    psi_t = H [ X'(R_t - mu_R) - X' e_t (z_t' lambda) + (0, z_t u_t) ]

The last term is the misspecification term; with e = 0 the variance of psi
reduces to Shanken's (1 + c) Omega + Sigma_f. Var(gamma-hat) is the
Newey-West long-run variance of psi_t divided by T. For the test of
lambda = E[f], the influence of mu_f, (f_t - mu_f), is subtracted from the
lambda block.
"""
import numpy as np

from . import fmb


def _estimate(R, F, w=None):
    """gamma = (X'X)^-1 X' mu_R with weighted moments (w sums to one)."""
    T = R.shape[0]
    w = np.full(T, 1.0 / T) if w is None else w
    mu_R = w @ R
    mu_f = w @ F
    Rc, Fc = R - mu_R, F - mu_f
    S_f = (Fc * w[:, None]).T @ Fc
    V_Rf = (Rc * w[:, None]).T @ Fc
    B = V_Rf @ np.linalg.inv(S_f)
    X = np.column_stack([np.ones(R.shape[1]), B])
    gamma = np.linalg.solve(X.T @ X, X.T @ mu_R)
    return gamma, mu_f


def influence(R, F):
    """(T x (K+1)) influence series psi_t of gamma = (gamma0, lambda')."""
    R = np.asarray(R, float)
    F = np.asarray(F, float)
    mu_R, mu_f = R.mean(0), F.mean(0)
    Rc, Fc = R - mu_R, F - mu_f
    S_f = Fc.T @ Fc / len(F)
    S_inv = np.linalg.inv(S_f)
    B = (Rc.T @ Fc / len(F)) @ S_inv
    X = np.column_stack([np.ones(R.shape[1]), B])
    H = np.linalg.inv(X.T @ X)
    gamma = H @ X.T @ mu_R
    lam = gamma[1:]
    e = mu_R - X @ gamma
    E = Rc - Fc @ B.T
    Z = Fc @ S_inv
    u = E @ e
    zl = Z @ lam
    core = Rc @ X - (E @ X) * zl[:, None]
    mis = np.column_stack([np.zeros(len(F)), Z * u[:, None]])
    return (core + mis) @ H.T, gamma, mu_f


def krs_standard_errors(R, F, nw_lags=None):
    """Estimates and misspecification-robust standard errors of gamma0,
    lambda and lambda - E[f]."""
    psi, gamma, mu_f = influence(R, F)
    V = fmb.newey_west_cov(psi, lags=nw_lags)
    Fc = np.asarray(F, float) - mu_f
    psi_d = psi[:, 1:] - Fc
    Vd = fmb.newey_west_cov(psi_d, lags=nw_lags)
    return dict(gamma=gamma, se_gamma=np.sqrt(np.diag(V)),
                diff=gamma[1:] - mu_f, se_diff=np.sqrt(np.diag(Vd)), T=len(F))
