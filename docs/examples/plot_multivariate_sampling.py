import numpy as np
import matplotlib.pyplot as plt
import aub_htp as ht
from aub_htp.random import BaseSpectralMeasureSampler

class ButterflySampler(BaseSpectralMeasureSampler):
    def sample(self, number_of_samples: int, random_state = None):
        p = np.random.rand(number_of_samples)
        theta = np.empty(number_of_samples)

        mask = p <= 0.5
        theta[mask] = np.random.uniform(-np.pi / 4, np.pi / 4, size=mask.sum())
        theta[~mask] = np.random.uniform(
            3 * np.pi / 4,
            5 * np.pi / 4,
            size=(~mask).sum()
        )

        x = np.cos(theta)
        y = np.sin(theta)

        return np.column_stack((x, y))

    def dimensions(self) -> int:
        return 2

    def mass(self) -> float:
        return 1.0


def main():
    alpha = 0.8
    size = 50000

    samples = ht.multivariate_alpha_stable.rvs(
        alpha=alpha,
        spectral_measure_sampler=ButterflySampler(),
        size=size,
        random_state=123
    )

    x = samples[:, 0]
    y = samples[:, 1]

    # Clip per dimension using percentiles
    x_low, x_high = -200, 200
    y_low, y_high = -200, 200

    mask = (
        (x >= x_low) & (x <= x_high) &
        (y >= y_low) & (y <= y_high)
    )

    x_clipped = x[mask]
    y_clipped = y[mask]

    plt.figure(figsize=(6, 6))
    plt.scatter(x_clipped, y_clipped, s=2, alpha=0.4)

    plt.title("Butterfly Alpha Stable Samples (Clipped)")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(alpha=0.3)
    plt.axis("equal")

    plt.tight_layout()
    plt.savefig("plot_multivariate_sampling.png")


if __name__ == "__main__":
    main()