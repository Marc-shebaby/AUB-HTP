# kmeans.py
"""
Heavy-tailed k-means clustering.

Uses α-location and α-power instead of mean and variance,
providing robust clustering for heavy-tailed data.
"""

import numpy as np

from aub_htp.alpha_stable_pdf.estimate import estimate_power, estimate_location

class KMeansHeavyTailed: #TODO: make excplicit the fact that it follows sklearn's BaseEstimator interface
    """
    Heavy-tailed k-means clustering.
    
    Uses α-location for centroids and α-power for inertia,
    providing robust clustering for heavy-tailed data.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    alpha : float
        Stability parameter. α=1 is Cauchy.
    random_state : int, default=37
        Random seed.
    convergence_tolerange : float, default=1e-6
        Convergence tolerance (relative change in inertia).
    max_itererations : int, default=100
        Maximum iterations.
    
    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.
    labels_ : ndarray of shape (n_samples,)
        Labels of each point from the last fit.
    inertia_ : float
        Global α-power (inertia) of the final clustering.
    cluster_powers_ : ndarray of shape (n_clusters,)
        α-power for each cluster.
    n_iter_ : int
        Number of iterations run.
    """
    
    def __init__(
        self, 
        n_clusters: int, 
        alpha: float, 
        random_state: int = 37,
        convergence_tolerance: float = 1e-6,
        max_itererations: int = 100,
    ):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.random_state = random_state
        self.convergence_tolerance = convergence_tolerance
        self.max_itererations = max_itererations

        self.is_fitted_ = False # Later for sklearn integration.

    def fit(self, X: np.ndarray, y=None) -> "KMeansHeavyTailed":
        """
        Fit the heavy-tailed k-means clustering.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ignored
            Not used, present for API consistency.
        
        Returns
        -------
        self
            Fitted estimator.
        """
        result = kmeans_heavy_tailed(
            X, 
            K=self.n_clusters, 
            alpha=self.alpha,
            tol=self.convergence_tolerance,
            max_iter=self.max_itererations,
        )
        
        self.cluster_centers_ = result["locations_mu"]
        self.labels_ = result["assignments"]
        self.inertia_ = result["global_power_I"]
        self.cluster_powers_ = result["cluster_power_P"]
        self.n_iter_ = result["iterations"]
        self.inertia_history_ = result["I_history"]

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the closest cluster for each sample.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New data to predict.
        
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        
        return assign_by_nearest_mu(X, self.cluster_centers_)
    
    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Fit and predict in one step.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ignored
            Not used, present for API consistency.
        
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        self.fit(X, y)
        return self.labels_

def init_locations_percentiles(x: np.ndarray, K: int) -> np.ndarray:
    """Initialize centroids using percentiles (Algorithm 2 in paper)."""
    x = np.asarray(x)
    if x.ndim == 1:
        y = np.array([np.percentile(x, 100.0*(2*i+1)/(2*K)) for i in range(K)], dtype=float)
        print(y.shape, " initlocationpercentiles")
        # TODO y = y.reshape(K, 1)   # shape (K, 1)

        return y
    else:
        # d-D: take percentiles along each dimension
        n, d = x.shape
        centroids = np.zeros((K, d), dtype=float)
        for dim in range(d):
            for i in range(K):
                centroids[i, dim] = np.percentile(x[:, dim], 100.0*(2*i+1)/(2*K))
        return centroids

def assign_by_nearest_mu(x, mus):
    x = np.atleast_2d(x)
    mus = np.atleast_2d(mus)
    dists = np.linalg.norm(x[:, None, :] - mus[None, :, :], axis=2)
    return np.argmin(dists, axis=1)


def update_locations_and_cluster_powers(x: np.ndarray,
                                        assignments: np.ndarray,
                                        K: int,
                                        alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute new cluster locations μ_j and cluster α-power P_j."""
    x = np.atleast_2d(x)
    n, d = x.shape
    mus = np.zeros((K, d), dtype=float)
    P_cluster = np.zeros(K, dtype=float)

    for j in range(K):
        xj = x[assignments == j]
        if xj.size == 0:
            continue  # empty cluster, keep previous centroid
        mu_j, _ = estimate_location(xj, alpha=alpha)
        P_j = estimate_power(xj - mu_j, alpha=alpha)
        if np.isscalar(mu_j):
            mus[j] = mu_j
        else:
            mus[j] = mu_j
        P_cluster[j] = P_j
    return mus, P_cluster

def global_power_I(x: np.ndarray, assignments: np.ndarray, mus: np.ndarray,
                   alpha: float) -> float:
    """Compute global inertia: P_alpha over all cluster residuals."""
    x = np.atleast_2d(x)
    mus = np.atleast_2d(mus)
    residuals = x - mus[assignments]
    return estimate_power(residuals, alpha=alpha)

def kmeans_heavy_tailed(x: np.ndarray,
                        K: int,
                        alpha: float = 1.0,
                        tol: float = 1e-6,
                        max_iter: int = 100) -> dict:
    """
    Heavy-tailed k-means clustering for 1-D or d-D data using location_L and P_alpha.
    Stops when relative change in global P_alpha (inertia) is ≤ tol.
    """


    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]  # Convert to shape (n,1) for 1-D points
    print(x.shape, " fugfiduagfigiadg")
    
    # Initialize centroids
    mus = init_locations_percentiles(x, K)
    assignments = assign_by_nearest_mu(x, mus)
    P_hist = []

    for it in range(max_iter):
        # Step 1: update μ_j and cluster P_j
        mus, P_cluster = update_locations_and_cluster_powers(x, assignments, K, alpha)

        # Step 2: compute global inertia
        I = global_power_I(x, assignments, mus, alpha)
        P_hist.append(I)

        # Step 3: assign points to nearest centroid
        new_assignments = assign_by_nearest_mu(x, mus)

        # Step 4: check convergence (relative change in inertia)
        if it >= 1:
            rel_change = abs(P_hist[-1] - P_hist[-2]) / max(P_hist[-2], 1e-12)
            if rel_change <= tol:
                assignments = new_assignments
                break

        assignments = new_assignments

    return {
        "assignments": assignments,
        "locations_mu": mus,
        "cluster_power_P": P_cluster,
        "global_power_I": I,
        "I_history": np.array(P_hist),
        "iterations": it + 1
    }
