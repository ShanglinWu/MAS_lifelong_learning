"""Multi-factor scoring retrieval for LLMA-Mem (Eq. 7-10 from paper)."""

import math
import time
from typing import List, Optional, Tuple, Union

import numpy as np

from .schemas import EpisodicRecord, ProceduralRecord


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def _jaccard_similarity(set_a: List[str], set_b: List[str]) -> float:
    """Compute Jaccard similarity between two string lists."""
    a = set(set_a)
    b = set(set_b)
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def _recency_score(
    record_timestamp: float,
    current_time: float,
    gamma: float = 0.01,
) -> float:
    """Recency = exp(-gamma * (t_current - t_m))."""
    dt = max(current_time - record_timestamp, 0.0)
    return math.exp(-gamma * dt)


def _relevance_score(
    record_embedding: List[float], query_embedding: List[float]
) -> float:
    """Relevance = cosine_similarity(embed(m), embed(q))."""
    return _cosine_similarity(record_embedding, query_embedding)


def _importance_score(record: Union[EpisodicRecord, ProceduralRecord]) -> float:
    """
    Importance(p) = success_rate * log(1 + viewed_times).
    For episodic records, use collaboration_quality as a proxy.
    """
    if isinstance(record, ProceduralRecord):
        total = record.success_count + record.failure_count
        success_rate = record.success_count / total if total > 0 else 0.5
        return success_rate * math.log(1 + record.viewed_times)
    else:
        # Episodic: use collaboration_quality as importance proxy
        return record.collaboration_quality * 0.5


def _team_match_score(
    record_team: List[str], query_team: List[str]
) -> float:
    """Team match = Jaccard similarity of team compositions."""
    return _jaccard_similarity(record_team, query_team)


def score_record(
    record: Union[EpisodicRecord, ProceduralRecord],
    query_embedding: List[float],
    query_team: List[str],
    current_time: Optional[float] = None,
    w_recency: float = 0.15,
    w_relevance: float = 0.55,
    w_importance: float = 0.2,
    w_team: float = 0.1,
    gamma: float = 0.003,
) -> float:
    """
    Multi-factor scoring per the paper:
    score = w_recency*recency + w_relevance*relevance + w_importance*importance + w_team*team_match
    """
    if current_time is None:
        current_time = time.time()

    # Get timestamp
    if isinstance(record, EpisodicRecord):
        ts = record.timestamp
        team = record.team_composition
        emb = record.embedding
    else:
        ts = record.timestamps[-1] if record.timestamps else 0.0
        team = []  # Procedural records don't have team composition directly
        emb = record.embedding

    recency = _recency_score(ts, current_time, gamma)
    relevance = _relevance_score(emb, query_embedding)
    importance = _importance_score(record)
    team_m = _team_match_score(team, query_team)

    return (
        w_recency * recency
        + w_relevance * relevance
        + w_importance * importance
        + w_team * team_m
    )


def retrieve_top_k(
    records: List[Union[EpisodicRecord, ProceduralRecord]],
    query_embedding: List[float],
    query_team: List[str],
    k: int = 5,
    current_time: Optional[float] = None,
) -> List[Tuple[Union[EpisodicRecord, ProceduralRecord], float]]:
    """
    Score all records and return top-k by multi-factor score.

    Returns list of (record, score) tuples sorted by score descending.
    """
    scored: List[Tuple[Union[EpisodicRecord, ProceduralRecord], float]] = []
    for record in records:
        s = score_record(record, query_embedding, query_team, current_time)
        scored.append((record, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
