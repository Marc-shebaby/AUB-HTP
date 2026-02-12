import numpy as np
from abc import abstractmethod, ABC

class BaseSpectralMeasureSampler(ABC):
    '''
    A Spectral Measure Sampler is as an interface which defines:
        - self.sample(): sampling algorithm (self.sample)
        - self.mass(): the mass of the spectral measure
        - self.dimensions(): the number of dimensions of the spectral measure
    '''

    @abstractmethod
    def sample(self) -> np.ndarray:
        pass

    @abstractmethod
    def mass(self) -> float: # TODO in meeting: Where is it used other than mixed?
        pass

    @abstractmethod
    def dimensions(self) -> int:
        pass


class IsotropicSampler(BaseSpectralMeasureSampler):

    def __init__(self,
        number_of_dimensions: int,
        total_mass: float
    ):
        self.number_of_dimensions = number_of_dimensions
        self._total_mass = total_mass

    def sample(self) -> np.ndarray:
        X = np.random.normal(size=self.number_of_dimensions)
        X /= np.linalg.norm(X)
        return X #TODO: ask wael * self._total_mass / surface_of_sphere

    def dimensions(self) -> int:
        return self.number_of_dimensions

    def mass(self) -> float:
        return self._total_mass


class EllipticSampler(BaseSpectralMeasureSampler):

    def __init__(self,
        number_of_dimensions: int,
        sigma: np.ndarray,
        number_of_convergence_terms: int = 100,
        total_mass: float | None = None,
        alpha: float | None = None,
    ):
        total_mass_specified = total_mass is not None
        estimation_parameters_specified = alpha is not None

        assert total_mass_specified or estimation_parameters_specified, "You need to specify either (total_mass) or (alpha)"

        self.number_of_convergence_terms = number_of_convergence_terms
        self.number_of_dimensions = number_of_dimensions
        self.sigma = np.asarray(sigma)

        self._total_mass = total_mass
        self._alpha = alpha

    def sample(self) -> np.ndarray:
        U = np.random.normal(size=self.number_of_dimensions)
        U /= np.linalg.norm(U)
        L = np.linalg.cholesky(self.sigma)
        return U @ L.T

    def mass(self) -> float:
        if self._total_mass is not None:
            return self._total_mass
        return self.__class__.estimate_total_mass(self.number_of_convergence_terms, self.number_of_dimensions, self._alpha, self.sigma)

    def dimensions(self) -> int:
        return self.number_of_dimensions

    @staticmethod
    def estimate_total_mass(
        number_of_convergence_terms: int,
        number_of_dimensions: int,
        alpha: float,
        sigma: np.ndarray
    ) -> float:
        U = np.random.normal(size=(number_of_convergence_terms, number_of_dimensions))
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        L = np.linalg.cholesky(sigma)
        norms = np.linalg.norm(U @ L.T, axis=1) ** alpha
        return float(np.mean(norms))


class DiscreteSampler(BaseSpectralMeasureSampler):
    def __init__(self,
        number_of_convergence_terms: int,
        positions: np.ndarray,
        weights: np.ndarray
    ):
        assert len(positions) == len(weights)
        self.number_of_convergence_terms = number_of_convergence_terms
        self.positions = np.asarray(positions)
        self.weights = np.asarray(weights)

    def sample(self) -> np.ndarray:
        return np.random.choice(self.positions, p = self.weights / self.weights.sum())

    def mass(self) -> float:
        return self.weights.sum()

    def dimensions(self) -> int:
        return self.number_of_dimensions


class MixedSampler(BaseSpectralMeasureSampler):
    def __init__(self,
        spectral_measures: list[BaseSpectralMeasureSampler],
        weights: np.ndarray,
    ):
        assert len(weights) == len(spectral_measures)
        assert len(spectral_measures) > 0
        assert all(sprectral_measure.dimensions() == spectral_measures[0].dimensions() for sprectral_measure in spectral_measures)

        self.number_of_dimensions = spectral_measures[0].dimensions()
        self.spectral_measures = spectral_measures
        self.weights = np.asarray(weights)

    def sample(self) -> np.ndarray:
        choice = np.random.choice(len(self.weights), p = self.weights / self.weights.sum())
        return self.spectral_measures[choice].sample()

    def mass(self) -> float: 
        return sum(
            spectral_measure.mass() * weight
            for spectral_measure, weight in zip(self.spectral_measures, self.weights)
        )

    def dimensions(self) -> int:
        return self.number_of_dimensions