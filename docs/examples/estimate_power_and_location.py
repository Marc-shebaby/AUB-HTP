import matplotlib.pyplot as plt
import numpy as np
from aub_htp import generate_alpha_stable_pdf


from aub_htp.alpha_stable_pdf.estimate.estimate_power import estimate_power
from aub_htp.alpha_stable_pdf.estimate.estimate_location import estimate_location

alpha = 2.0
beta = 0.
gamma = 1.
delta = 0.

x_vals = np.linspace(-100, 100, 4000)
y = generate_alpha_stable_pdf(x_vals, alpha, beta, gamma, delta)


plt.plot(x_vals, y, label="Custom Stable PDF", linewidth=2)
plt.title(f"Stable PDF (alpha={alpha}, beta={beta})")
plt.xlabel("x")
plt.ylabel("PDF")
plt.legend()
plt.grid(True)

plt.show()

weights = y / y.sum()  # normalize to probabilities
samples = np.random.choice(x_vals, size=1000, p=weights)


print("The mean is: ", samples.mean(axis=0))
print("The standard deviation is: ", samples.std())
print("The power is: ", estimate_power(samples, alpha))
print("The location is: ", estimate_location(samples, alpha))
