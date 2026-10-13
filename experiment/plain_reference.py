"""Plaintext oracle for the ANN ranking semantics; no privacy or GPU claims."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReferenceResult:
    selected_clusters: tuple[int, ...]
    candidate_ids: tuple[int, ...]
    bucket_winners: tuple[int, ...]
    topk_ids: tuple[int, ...]
    exact_topk_ids: tuple[int, ...]
    recall_at_k: float


def squared_l2(vectors, query):
    vectors = np.asarray(vectors, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    if vectors.ndim != 2 or query.ndim != 1 or vectors.shape[1] != query.shape[0]:
        raise ValueError("Query dimension must equal feature dimension")
    if not np.isfinite(vectors).all() or not np.isfinite(query).all():
        raise ValueError("Features and query must be finite")
    delta = vectors - query
    return np.sum(delta * delta, axis=1)


def evaluate(ids, vectors, labels, centroids, query, *, u, buckets, k, seed=0):
    ids = np.asarray(ids)
    vectors = np.asarray(vectors)
    labels = np.asarray(labels)
    centroids = np.asarray(centroids)
    if ids.ndim != 1 or labels.shape != ids.shape or len(ids) != len(vectors):
        raise ValueError("IDs, vectors and labels must align")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("IDs must be unique")
    count = len(centroids)
    if not (1 <= u <= count and 1 <= k <= buckets and buckets >= 1):
        raise ValueError("Invalid u, bucket or k configuration")
    if np.any(labels < 0) or np.any(labels >= count):
        raise ValueError("Invalid cluster label")

    # Stable plaintext choices are one permitted tie order; encrypted ties may differ.
    centroid_distances = squared_l2(centroids, query)
    selected = sorted(range(count), key=lambda c: centroid_distances[c])[:u]
    distances = squared_l2(vectors, query)
    indices = [i for i, label in enumerate(labels) if int(label) in selected]
    exact = sorted(range(len(ids)), key=lambda i: distances[i])[:k]

    # Public-seed balanced random layout, solely for a plaintext test oracle.
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(indices))
    winners = []
    for bucket in range(buckets):
        members = [indices[int(permutation[pos])] for pos in range(bucket, len(indices), buckets)]
        if members:
            winners.append(min(members, key=lambda i: distances[i]))
    ranked = sorted(winners, key=lambda i: distances[i])[:k]
    output = tuple(int(ids[i]) for i in ranked)
    exact_ids = tuple(int(ids[i]) for i in exact)
    return ReferenceResult(
        tuple(selected),
        tuple(int(ids[i]) for i in indices),
        tuple(int(ids[i]) for i in winners),
        output,
        exact_ids,
        len(set(output) & set(exact_ids)) / min(k, len(exact_ids)),
    )
