"""Consolidation pipeline for LLMA-Mem (Algorithm 1 from paper)."""

import uuid
import time
from typing import List, Optional

import numpy as np

from .episodic_memory import EpisodicMemory
from .procedural_memory import ProceduralMemory
from .retrieval import _cosine_similarity
from .schemas import EpisodicRecord, ProceduralRecord


def _cluster_episodes(
    episodes: List[EpisodicRecord],
    distance_threshold: float = 1.0,
) -> List[List[EpisodicRecord]]:
    """
    Cluster episodes by embedding similarity using AgglomerativeClustering.
    Falls back to single-cluster if embeddings are missing.
    """
    if not episodes:
        return []

    # Check if embeddings are available
    has_embeddings = all(len(ep.embedding) > 0 for ep in episodes)

    if not has_embeddings or len(episodes) < 2:
        return [episodes]

    try:
        from sklearn.cluster import AgglomerativeClustering

        embeddings = np.array([ep.embedding for ep in episodes])

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)

        clusters: dict = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(episodes[idx])

        return list(clusters.values())
    except Exception:
        # Fallback: treat all as one cluster
        return [episodes]


def _extract_procedure_from_cluster(
    cluster: List[EpisodicRecord],
    agent_id: str,
    min_successes: int = 3,
    min_success_rate: float = 0.6,
) -> Optional[ProceduralRecord]:
    """
    Extract a procedural record from a cluster of episodes.
    Requires >= min_successes and > min_success_rate.
    """
    if len(cluster) < min_successes:
        return None

    successes = sum(
        1 for ep in cluster if "success" in ep.outcome.lower()
    )
    total = len(cluster)
    success_rate = successes / total if total > 0 else 0.0

    if successes < min_successes or success_rate < min_success_rate:
        return None

    # Build procedure from cluster
    actions = []
    lessons = []
    source_ids = []
    for ep in cluster:
        actions.extend(ep.actions_taken)
        if ep.lessons_learned:
            lessons.append(ep.lessons_learned)
        source_ids.append(ep.episode_id)

    # Compute average embedding
    embeddings = [ep.embedding for ep in cluster if ep.embedding]
    avg_embedding = []
    if embeddings:
        avg_embedding = np.mean(embeddings, axis=0).tolist()

    unique_actions = list(dict.fromkeys(actions))[:5]
    knowledge = (
        f"Strategy derived from {total} episodes "
        f"(success rate: {success_rate:.0%}). "
        f"Key actions: {'; '.join(unique_actions)}. "
        f"Lessons: {'; '.join(lessons[:3])}"
    )

    return ProceduralRecord(
        agent_id=agent_id,
        procedure_id=str(uuid.uuid4()),
        timestamps=[time.time()],
        title=f"Procedure from {total} episodes",
        type="strategy",
        knowledge_content=knowledge,
        success_count=successes,
        failure_count=total - successes,
        source_episodes=source_ids,
        embedding=avg_embedding,
    )


def consolidate(
    episodic: EpisodicMemory,
    procedural: ProceduralMemory,
    agent_id: str,
    trigger_interval: int = 10,
    similarity_merge_threshold: float = 0.85,
    prune_min_confidence: float = 0.3,
    prune_min_unused_episodes: int = 5,
) -> int:
    """
    Run the consolidation pipeline (Algorithm 1):
    1. Cluster episodes by embedding similarity
    2. Extract procedures from clusters with sufficient success
    3. Merge similar procedures
    4. Prune low-confidence procedures

    Only triggers when episode count is a multiple of trigger_interval.
    Returns number of new procedures created.
    """
    episode_count = episodic.count()
    if episode_count == 0 or episode_count % trigger_interval != 0:
        return 0

    episodes = episodic.get_episodes_by_agent(agent_id)
    if not episodes:
        return 0

    # Step 1: Cluster
    clusters = _cluster_episodes(episodes)

    # Step 2: Extract procedures
    new_procedures: List[ProceduralRecord] = []
    for cluster in clusters:
        proc = _extract_procedure_from_cluster(cluster, agent_id)
        if proc is not None:
            new_procedures.append(proc)

    # Add new procedures
    for proc in new_procedures:
        procedural.add_procedure(proc)

    # Step 3: Merge similar procedures
    all_procs = procedural.get_all_procedures()
    merged = set()
    for i in range(len(all_procs)):
        if all_procs[i].procedure_id in merged:
            continue
        for j in range(i + 1, len(all_procs)):
            if all_procs[j].procedure_id in merged:
                continue
            if all_procs[i].embedding and all_procs[j].embedding:
                sim = _cosine_similarity(
                    all_procs[i].embedding, all_procs[j].embedding
                )
                if sim > similarity_merge_threshold:
                    result = procedural.merge_similar(
                        all_procs[i].procedure_id,
                        all_procs[j].procedure_id,
                    )
                    if result:
                        # Mark the removed one
                        removed_id = (
                            all_procs[j].procedure_id
                            if result.procedure_id == all_procs[i].procedure_id
                            else all_procs[i].procedure_id
                        )
                        merged.add(removed_id)

    # Step 4: Prune
    procedural.prune(
        min_confidence=prune_min_confidence,
        current_episode_count=episode_count,
        min_unused_episodes=prune_min_unused_episodes,
    )

    return len(new_procedures)
