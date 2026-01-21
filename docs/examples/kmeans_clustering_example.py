"""
Heavy-Tailed K-Means Clustering Example

This example demonstrates:
1. Sampling from 3 separate heavy-tailed distributions with different locations
2. Running HeavyTailedKMeans clustering
3. Computing the misclassification error rate
"""

import numpy as np
import matplotlib.pyplot as plt

from aub_htp import generate_alpha_stable_pdf
from aub_htp import KMeansHeavyTailed


def sample_from_stable_1d(n_samples: int, alpha: float, beta: float,
                          gamma: float, delta: float) -> np.ndarray:
    x_vals = np.linspace(delta - 50, delta + 50, 10000)
    y = generate_alpha_stable_pdf(x_vals, alpha, beta, gamma, delta)
    weights = y / y.sum()
    samples = np.random.choice(x_vals, size=n_samples, p=weights)
    return samples


def sample_from_stable(n_samples: int, alpha: float, beta: float,
                       gamma: float, delta: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            sample_from_stable_1d(n_samples, alpha, beta, gamma, delta_i)
            for delta_i in delta
        ],
        float
    ).transpose()


def compute_misclassification_rate(true_labels: np.ndarray,
                                   predicted_labels: np.ndarray,
                                   n_clusters: int) -> float:
    """
    Compute misclassification rate using best permutation matching.
    Since cluster labels are arbitrary, find the permutation that minimizes error.
    """
    from itertools import permutations

    best_error = 1.0
    for perm in permutations(range(n_clusters)):
        remapped = np.array([perm[label] for label in predicted_labels])
        error = np.mean(remapped != true_labels)
        best_error = min(best_error, error)

    return best_error


def main():
    np.random.seed(42)

    # Distribution parameters
    alpha = 1.0  # Cauchy distribution (heavy tails)
    beta = 0.0   # Symmetric
    gamma = 1.0  # Scale

    # 3 cluster centers in 2D
    cluster_centers = [
        np.array([-9.0, 0.0]),
        np.array([4.5, 7.5]),
        np.array([4.5, -7.5]),
    ]
    n_clusters = len(cluster_centers)
    n_samples_per_cluster = 150

    # Generate samples from each cluster
    X_list = []
    y_true_list = []

    for cluster_id, center in enumerate(cluster_centers):
        samples = sample_from_stable(
            n_samples=n_samples_per_cluster,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=center
        )
        X_list.append(samples)
        y_true_list.append(np.full(n_samples_per_cluster, cluster_id))

    X = np.vstack(X_list)
    y_true = np.concatenate(y_true_list)

    print(f"Generated {len(X)} samples from {n_clusters} clusters")
    print(f"Data shape: {X.shape}")
    print(f"Alpha (stability): {alpha}")
    print(f"Cluster centers: {cluster_centers}")
    print()

    # Run Heavy-Tailed K-Means
    kmeans = KMeansHeavyTailed(
        n_clusters=n_clusters,
        alpha=alpha,
        max_itererations=100,
        convergence_tolerance=1e-6,
    )
    kmeans.fit(X)

    print("=== Clustering Results ===")
    print(f"Estimated cluster centers:\n{kmeans.cluster_centers_}")
    print(f"Inertia (global α-power): {kmeans.inertia_:.4f}")
    print(f"Iterations: {kmeans.n_iter_}")
    print()

    # Compute misclassification rate
    y_pred = kmeans.labels_
    error_rate = compute_misclassification_rate(y_true, y_pred, n_clusters)
    accuracy = 1 - error_rate

    print("=== Classification Performance ===")
    print(f"Misclassification error rate: {error_rate:.2%}")
    print(f"Accuracy: {accuracy:.2%}")
    print()

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: True labels
    ax1 = axes[0]
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    for cluster_id in range(n_clusters):
        mask = y_true == cluster_id
        ax1.scatter(X[mask, 0], X[mask, 1], c=colors[cluster_id],
                    alpha=0.6, s=30, label=f'Cluster {cluster_id}')

    # Mark true centers
    for i, center in enumerate(cluster_centers):
        ax1.scatter(center[0], center[1], c='black', marker='x',
                    s=200, linewidths=3, zorder=5)

    ax1.set_title('Ground Truth Labels', fontsize=14)
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-25, 25)
    ax1.set_ylim(-25, 25)

    # Plot 2: Predicted labels
    ax2 = axes[1]
    for cluster_id in range(n_clusters):
        mask = y_pred == cluster_id
        ax2.scatter(X[mask, 0], X[mask, 1], c=colors[cluster_id],
                    alpha=0.6, s=30, label=f'Cluster {cluster_id}')

    # Mark estimated centers
    for i in range(n_clusters):
        center = kmeans.cluster_centers_[i]
        ax2.scatter(center[0], center[1], c='black', marker='*',
                    s=300, linewidths=2, zorder=5, edgecolors='white')

    ax2.set_title(f'HeavyTailedKMeans Predictions\n(Error Rate: {error_rate:.2%})',
                  fontsize=14)
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-25, 25)
    ax2.set_ylim(-25, 25)

    plt.tight_layout()
    plt.savefig('kmeans_clustering_result.png', dpi=150, bbox_inches='tight')
    print("Plot saved to 'kmeans_clustering_result.png'")
    plt.show()


if __name__ == "__main__":
    main()
