import numpy as np
import numpy.typing as npt
from typing import Literal
from scipy.stats import rv_continuous
from scipy._lib.doccer import inherit_docstring_from
from scipy.stats._multivariate import multi_rv_generic
from scipy.stats._distn_infrastructure import _ShapeInfo

from .pdf import generate_alpha_stable_pdf


class alpha_stable_gen(rv_continuous):
    _parameterization: Literal["S1", "S0"] = "S1"

    @property
    def parameterization(self):
        return self._parameterization

    @parameterization.setter
    def parameterization(self, parametrization: Literal["S1", "S0"]):
        if parametrization not in ("S1", "S0"):
            raise ValueError(f"Parametrization '{parametrization}' not supported. Use 'S1' (default) or 'S0'.")
        self._parameterization = parametrization

    def with_parametrization(self, parametrization: Literal["S1", "S0"]):
        self.parameterization = parametrization
        return self

    def _shape_info(self):
        return [_ShapeInfo("alpha", False, (0, 2), (False, True)),
                _ShapeInfo("beta", False, (-1, 1), (True, True))]

    @inherit_docstring_from(rv_continuous)
    def pdf(self, x, *args, **kwds):
        # override base class version to correct
        # location for S1 parameterization
        (alpha, beta), delta, gamma = self._parse_args(*args, **kwds)

        if self.parameterization == "S0" or self.parameterization == "S1" and np.all(alpha != 1): #TODO: multiple but different alpha?
            return super().pdf(x, *args, **kwds)

        elif self.parameterization == "S1":
            _kwds = kwds.copy()
            _kwds.pop("loc", None)
            return super().pdf(x, *args,
                               loc = delta + (2 / np.pi) * beta * gamma * np.log(gamma), # Fix location
                               **_kwds)
        else:
            raise AssertionError("Unknown parametrization type")

    def _pdf(self, x, alpha, beta):
        x = np.asarray(x).ravel()
        alpha = np.asarray(alpha).ravel()
        beta = np.asarray(beta).ravel()

        if np.all(alpha == alpha[0]) and np.all(beta == beta[0]) and np.all(np.diff(x)>0):

            # -1/2 and 1/2 have been chosen as arbitrary "close" values to x for interpolation to work.
            padded_x = np.concatenate([[x[0] - 1/2], x, [x[-1] + 1/2]])
            padded_pdf = generate_alpha_stable_pdf(padded_x, alpha[0], beta[0], 1, 0)
            return padded_pdf[1:-1]

        return np.array([
            generate_alpha_stable_pdf([xi - 1/2, xi, xi + 1/2], ai, bi, 1, 0)[1]
            for xi, ai, bi in zip(x, alpha, beta)
        ])

    def _argcheck(self, alpha, beta):
        return (0 < alpha) & (alpha <= 2) & (-1 <= beta) & (beta <= 1)


    def _rvs(self, alpha, beta, size, random_state):
        raise NotImplementedError()


class multi_variate_alpha_stable(multi_rv_generic):
    pass