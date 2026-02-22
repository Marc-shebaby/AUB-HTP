import numpy as np
import matplotlib.pyplot as plt
import aub_htp as ht

def main():
    alpha = 0.4
    beta = 0.8
    loc = 2
    scale = 3

    size = 100000

    ht.alpha_stable.with_parametrization("S0")

    samples = ht.alpha_stable.rvs(
        alpha=alpha,
        beta=beta,
        loc=loc,
        scale=scale,
        size=size,
        random_state=42
    )

    # Clip using percentiles (robust for heavy tails)
    lower, upper = -20, 20
    clipped = samples[(samples >= lower) & (samples <= upper)]

    plt.figure(figsize=(8, 5))
    plt.hist(clipped, bins=200, density=True)

    plt.title("Univariate Alpha Stable Sampling, alpha = 0.4, beta = 0.8, loc = 2, scale = 3, (Clipped)")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("plot_univariate_sampling.png")


if __name__ == "__main__":
    main()