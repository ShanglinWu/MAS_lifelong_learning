import math
from typing import Iterable, List


def cosine_similarity(vector_a: Iterable[float], vector_b: Iterable[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = list(vector_a)
    b = list(vector_b)
    if len(a) != len(b) or not a:
        return 0.0

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def cluster_by_similarity(
    embeddings: List[List[float]],
    min_similarity: float = 0.72,
) -> List[List[int]]:
    """
    Group embeddings into connected components based on a similarity threshold.

    This is a lightweight replacement for agglomerative clustering and keeps the
    repo usable without heavy numerical dependencies.
    """
    if not embeddings:
        return []

    visited = [False] * len(embeddings)
    clusters: List[List[int]] = []

    for start_index in range(len(embeddings)):
        if visited[start_index]:
            continue

        queue = [start_index]
        visited[start_index] = True
        cluster = [start_index]

        while queue:
            current = queue.pop()
            for candidate in range(len(embeddings)):
                if visited[candidate]:
                    continue
                score = cosine_similarity(embeddings[current], embeddings[candidate])
                if score >= min_similarity:
                    visited[candidate] = True
                    queue.append(candidate)
                    cluster.append(candidate)

        clusters.append(cluster)

    return clusters
