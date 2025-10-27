"""
Swarm Intelligence clustering using Flock by Leader algorithm for breast cancer classification.

Author: Gokul Ganesan
Project: Clustering CNN Features for Breast Cancer Prediction
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlockByLeader:
    """
    Flock by Leader clustering algorithm inspired by swarm intelligence.

    This algorithm treats each data point as a virtual "bird" and assigns
    clusters ("flocks") based on neighbor proximity using biologically-inspired
    flocking behavior.
    """

    def __init__(self, max_distance=0.1, min_samples=40, random_state=42):
        """
        Initialize the Flock by Leader clustering algorithm.

        Args:
            max_distance (float): Maximum distance for neighbors to join a flock
            min_samples (int): Minimum number of samples to form a core flock
            random_state (int): Random seed for reproducibility
        """
        self.max_distance = max_distance
        self.min_samples = min_samples
        self.random_state = random_state
        self.labels_ = None
        self.n_clusters_ = 0
        self.leaders_ = []

    def fit(self, X):
        """
        Perform Flock by Leader clustering.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)

        Returns:
            self
        """
        np.random.seed(self.random_state)
        n_samples = X.shape[0]

        # Initialize labels as unassigned (-1)
        self.labels_ = np.full(n_samples, -1, dtype=int)

        # Track visited points
        visited = np.zeros(n_samples, dtype=bool)

        # Compute pairwise distances
        logger.info("Computing pairwise distances...")
        distances = euclidean_distances(X)

        # Find potential leaders (points with many neighbors)
        neighbors_count = np.sum(distances <= self.max_distance, axis=1)

        # Sort points by number of neighbors (descending)
        potential_leaders = np.argsort(-neighbors_count)

        current_cluster = 0

        logger.info(f"Starting flocking with max_distance={self.max_distance}, min_samples={self.min_samples}")

        # Iterate through potential leaders
        for leader_idx in potential_leaders:
            if visited[leader_idx]:
                continue

            # Find neighbors within max_distance
            neighbor_mask = distances[leader_idx] <= self.max_distance
            neighbor_indices = np.where(neighbor_mask)[0]

            # Only form a flock if there are enough neighbors
            if len(neighbor_indices) >= self.min_samples:
                # Assign all neighbors to this flock
                for idx in neighbor_indices:
                    if not visited[idx]:
                        self.labels_[idx] = current_cluster
                        visited[idx] = True

                self.leaders_.append(leader_idx)
                current_cluster += 1

        # Assign remaining unvisited points to nearest cluster
        unassigned = np.where(self.labels_ == -1)[0]
        if len(unassigned) > 0 and current_cluster > 0:
            logger.info(f"Assigning {len(unassigned)} unassigned points to nearest flock...")

            # For each unassigned point, find the nearest assigned point
            for idx in unassigned:
                # Find distances to all assigned points
                assigned_mask = self.labels_ != -1
                if np.any(assigned_mask):
                    assigned_indices = np.where(assigned_mask)[0]
                    dists_to_assigned = distances[idx, assigned_indices]
                    nearest_assigned_idx = assigned_indices[np.argmin(dists_to_assigned)]
                    self.labels_[idx] = self.labels_[nearest_assigned_idx]

        self.n_clusters_ = current_cluster
        logger.info(f"Flocking completed: {self.n_clusters_} flocks formed")

        return self

    def fit_predict(self, X):
        """
        Perform clustering and return labels.

        Args:
            X (np.ndarray): Feature matrix

        Returns:
            np.ndarray: Cluster labels
        """
        self.fit(X)
        return self.labels_

    def get_flock_centers(self, X):
        """
        Get the center (mean) of each flock.

        Args:
            X (np.ndarray): Feature matrix

        Returns:
            np.ndarray: Flock centers
        """
        if self.labels_ is None:
            raise ValueError("Model must be fitted first")

        centers = []
        for cluster_id in range(self.n_clusters_):
            cluster_mask = self.labels_ == cluster_id
            if np.any(cluster_mask):
                center = X[cluster_mask].mean(axis=0)
                centers.append(center)

        return np.array(centers)


class SwarmClusterer:
    """
    Main clustering interface supporting both traditional and swarm-based methods.
    """

    def __init__(self, method='flock', n_clusters=2, **kwargs):
        """
        Initialize the swarm clusterer.

        Args:
            method (str): Clustering method ('kmeans' or 'flock')
            n_clusters (int): Number of clusters (used for KMeans)
            **kwargs: Additional parameters for the clustering algorithm
        """
        self.method = method
        self.n_clusters = n_clusters
        self.kwargs = kwargs
        self.model = None
        self.labels_ = None

    def fit(self, X):
        """
        Fit the clustering model.

        Args:
            X (np.ndarray): Feature matrix

        Returns:
            self
        """
        logger.info(f"Fitting {self.method} clustering...")

        if self.method == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.kwargs.get('random_state', 42),
                n_init=10,
                max_iter=300
            )
            self.labels_ = self.model.fit_predict(X)

        elif self.method == 'flock':
            self.model = FlockByLeader(
                max_distance=self.kwargs.get('max_distance', 0.1),
                min_samples=self.kwargs.get('min_samples', 40),
                random_state=self.kwargs.get('random_state', 42)
            )
            self.labels_ = self.model.fit_predict(X)
            self.n_clusters = self.model.n_clusters_

        else:
            raise ValueError(f"Unknown method: {self.method}")

        logger.info(f"Clustering completed with {self.n_clusters} clusters")
        return self

    def fit_predict(self, X):
        """
        Fit the model and return cluster labels.

        Args:
            X (np.ndarray): Feature matrix

        Returns:
            np.ndarray: Cluster labels
        """
        self.fit(X)
        return self.labels_

    def predict(self, X):
        """
        Predict cluster labels for new data.

        Args:
            X (np.ndarray): Feature matrix

        Returns:
            np.ndarray: Predicted cluster labels
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")

        if self.method == 'kmeans':
            return self.model.predict(X)
        elif self.method == 'flock':
            # For flock, assign to nearest cluster center
            centers = self.model.get_flock_centers(X)
            distances = cdist(X, centers, metric='euclidean')
            return np.argmin(distances, axis=1)

    def get_cluster_centers(self, X):
        """
        Get cluster centers.

        Args:
            X (np.ndarray): Feature matrix (needed for flock method)

        Returns:
            np.ndarray: Cluster centers
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")

        if self.method == 'kmeans':
            return self.model.cluster_centers_
        elif self.method == 'flock':
            return self.model.get_flock_centers(X)


def hyperparameter_search(X, param_grid, n_trials=10):
    """
    Perform Bayesian-style hyperparameter search for Flock by Leader.

    Args:
        X (np.ndarray): Feature matrix
        param_grid (dict): Dictionary with 'max_distance' and 'min_samples' ranges
        n_trials (int): Number of random trials

    Returns:
        dict: Best parameters found
    """
    np.random.seed(42)

    best_params = None
    best_score = -np.inf

    logger.info(f"Starting hyperparameter search with {n_trials} trials...")

    for trial in range(n_trials):
        # Sample parameters
        max_distance = np.random.uniform(
            param_grid['max_distance'][0],
            param_grid['max_distance'][1]
        )
        min_samples = np.random.randint(
            param_grid['min_samples'][0],
            param_grid['min_samples'][1]
        )

        # Fit model
        clusterer = SwarmClusterer(
            method='flock',
            max_distance=max_distance,
            min_samples=min_samples
        )
        clusterer.fit(X)

        # Score based on number of clusters and silhouette score
        n_clusters = clusterer.n_clusters

        # Simple scoring: prefer reasonable number of clusters (2-10)
        if 2 <= n_clusters <= 10:
            score = 10.0 - abs(n_clusters - 5)  # Prefer around 5 clusters
        else:
            score = -abs(n_clusters - 5)

        logger.info(f"Trial {trial + 1}/{n_trials}: "
                   f"max_distance={max_distance:.4f}, "
                   f"min_samples={min_samples}, "
                   f"n_clusters={n_clusters}, "
                   f"score={score:.2f}")

        if score > best_score:
            best_score = score
            best_params = {
                'max_distance': max_distance,
                'min_samples': min_samples,
                'n_clusters': n_clusters
            }

    logger.info(f"Best parameters: {best_params}")
    return best_params


def main():
    """Test the clustering algorithms."""
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(1000, 50)

    # Test KMeans
    logger.info("\n=== Testing KMeans ===")
    kmeans_clusterer = SwarmClusterer(method='kmeans', n_clusters=2)
    kmeans_labels = kmeans_clusterer.fit_predict(X)
    logger.info(f"KMeans clusters: {np.unique(kmeans_labels)}")

    # Test Flock by Leader
    logger.info("\n=== Testing Flock by Leader ===")
    flock_clusterer = SwarmClusterer(
        method='flock',
        max_distance=0.1,
        min_samples=40
    )
    flock_labels = flock_clusterer.fit_predict(X)
    logger.info(f"Flock clusters: {np.unique(flock_labels)}")

    # Test hyperparameter search
    logger.info("\n=== Testing Hyperparameter Search ===")
    param_grid = {
        'max_distance': [0.05, 0.2],
        'min_samples': [20, 60]
    }
    best_params = hyperparameter_search(X, param_grid, n_trials=5)
    logger.info(f"Best parameters found: {best_params}")


if __name__ == '__main__':
    main()
