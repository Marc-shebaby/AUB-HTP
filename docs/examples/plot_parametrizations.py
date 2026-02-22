import numpy as np
import matplotlib.pyplot as plt
import aub_htp as ht

def main():
    x = np.linspace(-20, 20, 2000)

    alpha = 1
    beta = 0.8
    loc = 2
    scale = 3

    # Default S1
    ht.alpha_stable.with_parametrization("S1")
    y_s1 = ht.alpha_stable.pdf(x, alpha=alpha, beta=beta, loc=loc, scale=scale)

    # S0 parametrization
    ht.alpha_stable.with_parametrization("S0")
    y_s0 = ht.alpha_stable.pdf(x, alpha=alpha, beta=beta, loc=loc, scale=scale)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y_s1, label="S1 parametrization")
    plt.plot(x, y_s0, label="S0 parametrization", linestyle="--")

    plt.title("Alpha Stable PDF: S1 vs S0 Parametrizations, alpha = 1, beta = 0.8, loc = 2, scale = 3")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("plot_parametrizations.png")


if __name__ == "__main__":
    main()