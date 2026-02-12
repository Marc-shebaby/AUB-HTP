import matplotlib.pyplot as plt
import numpy as np
from aub_htp.random.spectral_measure_sampler import IsotropicSampler, EllipticSampler, MixedSampler
from aub_htp.random.alpha_stable_sampler import sample_alpha_stable_vector

isotropic_sampler = IsotropicSampler(2, 1)
x = np.asarray([sample_alpha_stable_vector(100, 1.2, isotropic_sampler) for _ in range(200)])

plt.scatter(x.T[0], x.T[1])

# Add title and labels (optional)
plt.title("Simple Scatter Plot")
plt.xlabel("X Value")
plt.ylabel("Y Value")

# Display the plot
plt.show()