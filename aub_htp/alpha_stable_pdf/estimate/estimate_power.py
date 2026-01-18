import numpy as np
from typing import Callable

from scipy.optimize import fsolve, brentq
from scipy.stats import levy_stable
from scipy import special
import json
from pathlib import Path

lookup_table_entropy: dict[float, float] | None = None

def _load_entropy_lookup() -> dict[float, float]:
    global lookup_table_entropy
    if lookup_table_entropy is None:
        with open(Path(__file__).parent / "lookup_table_entropy.json", "r") as f:
            lookup_table_entropy = json.load(f)
    return lookup_table_entropy

def _solve_for_function_equal_zero(
    function: Callable[[float], float],
    *,
    P0: float = 1,
    P_min: float = 1e-12,
    P_max: float = 1e12,
) -> float:
    """
    Solve fun(P) = 0 for P > 0 robustly using a known bracket.
    If no sign change is detected, fall back to fsolve in log-space.
    """
    # Step 1: Try using P_min and P_max directly as the bracket for brentq
    a, b = P_min, P_max
    fa, fb = function(a), function(b)

    # If we have a valid sign change, use brentq
    if np.sign(fa) != np.sign(fb):
        return brentq(lambda p: function(float(p)), a=a, b=b)

    # Step 2: Fall back to fsolve (log-space)
    # Using log-space to avoid numerical issues with very large or small P
    def log_fun(s: float) -> float:
        return function(np.exp(s))  # fun operates on P, but we use s = log(P)

    # Starting guess in log space
    s0 = np.log(P0)
    s_sol = fsolve(log_fun, s0)[0]  # Return the solution back to P
    return np.exp(s_sol)

def _h_Z_tilde_1d(alpha: float, gamma_ref: float) -> float:
    h_lookup = _load_entropy_lookup()#TODO: what if alpha is not in the lookup table? should we look for the closest value or terminate the program?
    #TODO: what if alpha is not in the lookup table? should we look for the closest value or terminate the program?
    return float(h_lookup[alpha]) + np.log(gamma_ref) 

def _neglogpdf_alpha_1d(z: np.ndarray, alpha: float, gamma_ref: float) -> np.ndarray:
    """
    Per-coordinate 1-D kernel using logpdf (avoids underflow and preserves scaling).
    Vectorized over z with any shape.
    """
    return -levy_stable.logpdf(z, alpha, 0, loc=0, scale=gamma_ref) 

def _cauchy_isotropic_constant(d: int) -> float:
    return float(special.digamma((d + 1) / 2) + np.log(4.0) - special.digamma(1.0))

def _estimate_power_1d(a: np.ndarray, alpha: float) -> float:
    if alpha == 1.0:
        h_cauchy = np.log(4.0)
        def g(P: float) -> float:
            z = a / P
            return np.mean(np.log(1.0 + z**2)) - h_cauchy
        return float(_solve_for_function_equal_zero(g))
    else:
        gamma_ref = (1.0 / alpha) ** (1.0 / alpha)
        h_ref = _h_Z_tilde_1d(alpha, gamma_ref)
        def f(P: float) -> float:
            z = a / P
            return (np.mean(_neglogpdf_alpha_1d(z, alpha, gamma_ref)) - h_ref)
        return float(_solve_for_function_equal_zero(f))

def _estimate_power_multivariate(a: np.ndarray, alpha: float) -> float:
    n, d = a.shape
    assert d > 1, "d must be greater than 1 for multivariate data."
    if alpha != 1.0:
        raise NotImplementedError("α≠1 not implemented (no simple multivariate SαS pdf).")
    C_d = _cauchy_isotropic_constant(d)
    norms = np.linalg.norm(a, axis=1)
    def g(P: float) -> float:
        z = norms / P
        return np.mean(np.log(1.0 + z**2)) - C_d
    return float(_solve_for_function_equal_zero(g))

def estimate_power(
    a: np.ndarray,
    alpha: float,
) -> float:
    """
    Estimate the power P_alpha(X) of data in array a sampled from X ~ S(alpha, beta, gamma, delta)
    
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
    P_star : float
        Power at the optimal location.
    """

    alpha = float(alpha)
    if not (0 < alpha <= 2):
        raise ValueError("alpha must be in (0, 2].")
    a = np.asarray(a, dtype=float)

    if a.ndim == 1 or (a.ndim == 2 and a.shape[1] == 1):
        return _estimate_power_1d(a.ravel(), alpha)

    elif a.ndim == 2:
        return _estimate_power_multivariate(a, alpha)
    
    else:
        raise ValueError("a must be 1-D (n,) or 2-D (n,d).")