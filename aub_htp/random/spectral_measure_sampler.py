import numpy as np
from scipy.special import gamma
from abc import abstractmethod, ABC

class BaseSpectralMeasureSampler(ABC):
    '''
    A Spectral Measure Sampler is as an interface which defines:
        - self.sample(): sampling algorithm (self.sample)
        - self.dimensions(): the number of dimensions of the spectral measure
    '''

    @abstractmethod
    def sample(self, number_of_samples: int) -> np.ndarray:
        pass

    @abstractmethod
    def dimensions(self) -> int:
        pass

    @abstractmethod
    def mass(self) -> float:
        pass


class IsotropicSampler(BaseSpectralMeasureSampler):

    def __init__(self,
        number_of_dimensions: int,
        alpha: float,
        gamma: float,
    ):
        self.number_of_dimensions = number_of_dimensions
        self.alpha = alpha
        self.gamma = gamma

    def sample(self, number_of_samples: int) -> np.ndarray:
        X = np.random.normal(size=(number_of_samples, self.number_of_dimensions))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X * self.gamma #TODO: fix

    def dimensions(self) -> int:
        return self.number_of_dimensions

    def mass(self) -> float:
        return 1 #TODO: fix

class EllipticSampler(BaseSpectralMeasureSampler):

    def __init__(self,
        number_of_dimensions: int,
        alpha: float,
        sigma: np.ndarray,
        mass: float | None = None,
    ):
        self.number_of_dimensions = number_of_dimensions
        self.alpha = alpha
        self.sigma = np.asarray(sigma)
        self._mass = mass or self._estimate_mass()

    def sample(self, number_of_samples: int) -> np.ndarray:
        X = np.random.normal(size=(number_of_samples, self.number_of_dimensions))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        L = np.linalg.cholesky(self.sigma)
        return X @ L.T

    def dimensions(self) -> int:
        return self.number_of_dimensions

    def mass(self) -> float:
        return self._mass

    def _estimate_mass(self, number_of_samples_taken_for_accuracy: int = 100):
        U = np.random.normal(size=(number_of_samples_taken_for_accuracy, self.dimentions()))
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        L = np.linalg.cholesky(self.sigma)
        norms = np.linalg.norm(U @ L.T, axis=1) ** self.alpha
        return np.mean(norms)


class DiscreteSampler(BaseSpectralMeasureSampler):

    def __init__(self,
        positions: np.ndarray,
        weights: np.ndarray
    ):
        self.positions = np.asarray(positions)
        self.weights = np.asarray(weights)
        assert self.positions.shape[0] == self.weights.shape[0] and self.positions.shape[0] > 0
        self.number_of_dimensions = self.positions.shape[1]
        self._mass = self.weights.sum()

    def sample(self, number_of_samples: int) -> np.ndarray:
        indices = np.random.choice(len(self.weights), size=number_of_samples, p=self.weights / self.weights.sum())
        return self.positions[indices]

    def dimensions(self) -> int:
        return self.number_of_dimensions

    def mass(self) -> float:
        return self._mass


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
        self._mass = self._calculate_mass()

    def sample(self, number_of_samples: int) -> np.ndarray:
        weights = self.weights / self.weights.sum()
        indices = np.random.choice(len(weights), size=number_of_samples, p=weights)

        samples = []
        for i in range(len(weights)):
            count = np.sum(indices == i)
            if count > 0:
                samples.append(self.spectral_measures[i].sample(count))

        return np.vstack(samples)

    def dimensions(self) -> int:
        return self.number_of_dimensions

    def mass(self) -> float:
        return float(self._mass)

    def _calculate_mass(self):
        return np.mean(
            spectral_measure.mass() * weight
                for spectral_measure, weight in zip(self.spectral_measures, self.weights)
        )

class UnivariateSampler(BaseSpectralMeasureSampler):

    def __init__(self,
        alpha: float,
        beta: float,
    ):
        self.alpha = alpha
        self.beta = beta

    def sample(self, number_of_samples: int) -> np.ndarray:
        p_plus = (1.0 + self.beta) / 2.0
        signs = np.where(
            np.random.rand(number_of_samples) <= p_plus,
            1.0,
            -1.0
        ).reshape(-1, 1) # reshape to (n, 1) since we expect vectors
        return signs

    def dimensions(self) -> int:
        return 1

    def mass(self) -> float:
        return 1 #TODO: fix