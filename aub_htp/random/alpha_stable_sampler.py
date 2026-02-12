import numpy as np
from scipy.special import gamma
from .spectral_measure_sampler import BaseSpectralMeasureSampler

def sample_alpha_stable_vector(
    number_of_convergence_terms: int,
    alpha: float,
    spectral_measure: BaseSpectralMeasureSampler,
):
    x = np.zeros(spectral_measure.dimensions())
    cumulative_exponential = 0

    for _ in range(number_of_convergence_terms):
        cumulative_exponential += np.random.exponential(1.0)
        x += spectral_measure.sample() * cumulative_exponential ** (-1 / alpha)

    x *= _c(alpha)
    return x

def _c(alpha):
    return _kappa(alpha) ** (-1 / alpha)

def _kappa(alpha):
    if abs(alpha - 1.0) < 1e-12:
        return np.pi / 2
    return gamma(2 - alpha) * np.cos(np.pi * alpha / 2) / (1 - alpha)
