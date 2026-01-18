import numpy as np
from .estimate_power import estimate_power
from scipy.optimize import minimize


def estimate_location(
    a: np.ndarray,
    alpha: float,
    *,
    maxiter: int = 2000,
) -> np.ArrayLike:
    """
    Estimate the location L_X = argmin_μ P_alpha(X - μ) for 1-D and d-D data a 
    sampled from X ~ S(alpha, beta, gamma, delta).
    for any #TODO: Validate that below is correct.
        - alpha in (0, 2]
        - beta in [-1, 1]
        - gamma in (0, ∞)
        - delta in (-∞, ∞)

    Parameters
    ----------
    a : np.ndarray
        Data array. Shape (n,) for 1-D or (n, d) for d-D.
    alpha : float
        Stability parameter in (0, 2].

    Returns
    -------
    mu_star : np.ndarray or float
        Optimal location estimate.
    """
    a = np.asarray(a, dtype=float)
    
    alpha = float(alpha)
    if not (0 < alpha <= 2):
        raise ValueError("alpha must be in (0, 2].")

    if a.ndim == 1:
        a = a[:, None]

    if a.ndim != 2:
        raise ValueError("X must be shape (n,) or (n,d)")

    n, d = a.shape

    res = minimize(
        lambda mu_vec: estimate_power(a - mu_vec, alpha=alpha), 
        x0=np.median(a, axis=0), 
        method="Powell", 
        options={"maxiter":maxiter}
    )
    mu_star = np.array(res.x, float)
    return mu_star
