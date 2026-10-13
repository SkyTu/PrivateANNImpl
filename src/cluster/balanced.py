"""Small-data balanced k-means reference for validating later implementations.

SciPy's linear_sum_assignment assigns each record to one of C*capacity centroid
slots. Its dense cost matrix is O(N^2), so this is a correctness oracle only.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class BalancedClusters:
    kept_ids: np.ndarray
    dropped_ids: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    centroids: np.ndarray


def balanced_kmeans(ids, features, clusters, *, seed=0, max_iter=30):
    ids = np.asarray(ids)
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or ids.ndim != 1 or len(ids) != len(features):
        raise ValueError("Expected one feature vector per ID")
    if not np.isfinite(features).all() or len(np.unique(ids)) != len(ids):
        raise ValueError("Features must be finite and IDs unique")
    if not isinstance(clusters, int) or clusters < 1 or clusters > len(ids):
        raise ValueError("Expected 1 <= clusters <= number of records")
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be positive")

    capacity = len(ids) // clusters
    count = capacity * clusters
    if count > 4096:
        raise ValueError("Dense reference is limited to 4096 retained records")
    # Stable input-order truncation makes dropped IDs auditable.
    kept_ids, dropped_ids = ids[:count].copy(), ids[count:].copy()
    vectors = features[:count].copy()
    rng = np.random.default_rng(seed)
    centers = vectors[rng.choice(count, size=clusters, replace=False)].copy()
    labels = None
    for _ in range(max_iter):
        costs = np.repeat(cdist(vectors, centers, metric="sqeuclidean"), capacity, axis=1)
        rows, slots = linear_sum_assignment(costs)
        next_labels = np.empty(count, dtype=np.int64)
        next_labels[rows] = slots // capacity
        if labels is not None and np.array_equal(labels, next_labels):
            break
        labels = next_labels
        centers = np.vstack([vectors[labels == c].mean(axis=0) for c in range(clusters)])
    return BalancedClusters(kept_ids, dropped_ids, vectors, labels, centers)
