"""
A-Mem style memory adapter for MARBLE.

This module provides an opt-in baseline memory backend inspired by the
Agentic Memory repository while preserving MARBLE's existing memory API.
It reuses MARBLE's configured LLM and embedding stack instead of bringing in
separate model providers.
"""

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from marble.llms.model_prompting import model_prompting
from marble.llms.text_embedding import text_embedding
from marble.memory.base_memory import BaseMemory


@dataclass
class AMemNote:
    """Single A-Mem style note with lightweight relationship metadata."""

    note_id: str
    content: str
    keywords: List[str] = field(default_factory=list)
    context: str = "General"
    tags: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S")
    )
    source: str = ""


class AMemMemory(BaseMemory):
    """
    A-Mem style local memory for a single MARBLE agent.

    The class keeps the same externally visible hooks MARBLE already relies on:
    `update`, `retrieve_latest`, `retrieve_all`, `get_memory_str`, and
    `set_task_context`.
    """

    EMBEDDING_MODEL = "bedrock/amazon.titan-embed-text-v2:0"
    ANALYSIS_FALLBACK_KEYWORD_LIMIT = 6
    MAX_EMBEDDING_CHARS = 20000
    EMBEDDING_TRUNCATION_MARKER = "\n...[truncated for embedding]...\n"

    def __init__(
        self,
        agent_id: str,
        llm_model: str = "",
        embedding_model: str = "",
        retrieval_top_k: int = 3,
        max_memory_context: int = 5,
        link_threshold: float = 0.72,
    ) -> None:
        super().__init__()
        self.agent_id = agent_id
        self.llm_model = llm_model
        self.embedding_model = embedding_model or self.EMBEDDING_MODEL
        self.retrieval_top_k = retrieval_top_k
        self.max_memory_context = max_memory_context
        self.link_threshold = link_threshold
        self.storage: List[Dict[str, Any]] = []
        self._current_task: str = ""
        self._current_agent_profile: str = ""

    def set_task_context(self, task: str, agent_profile: str = "") -> None:
        self._current_task = task
        self._current_agent_profile = agent_profile

    def _extract_json_object(self, raw_text: str) -> Dict[str, Any]:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in analysis response.")
        return json.loads(match.group(0))

    def _fallback_analysis(self, content: str) -> Dict[str, Any]:
        words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", content.lower())
        stop_words = {
            "the", "and", "for", "with", "that", "this", "from", "have", "has",
            "into", "your", "agent", "result", "action", "task", "call", "args",
        }
        keywords: List[str] = []
        for word in words:
            if word in stop_words or word in keywords:
                continue
            keywords.append(word)
            if len(keywords) >= self.ANALYSIS_FALLBACK_KEYWORD_LIMIT:
                break

        trimmed = re.sub(r"\s+", " ", content).strip()
        context = trimmed[:160] if trimmed else "General"
        tags = keywords[:3]
        return {
            "keywords": keywords,
            "context": context or "General",
            "tags": tags,
        }

    def _analyze_content(self, content: str) -> Dict[str, Any]:
        if not self.llm_model:
            return self._fallback_analysis(content)

        prompt = (
            "Analyze the memory content and return strict JSON with keys "
            "`keywords`, `context`, and `tags`.\n"
            "Rules:\n"
            "- `keywords`: short list of salient phrases\n"
            "- `context`: one concise sentence\n"
            "- `tags`: broad categories\n"
            "- return JSON only\n\n"
            f"Memory content:\n{content}"
        )
        try:
            response = model_prompting(
                llm_model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You analyze agent memories and return strict JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                return_num=1,
                max_token_num=256,
                temperature=0.0,
                top_p=None,
                stream=None,
            )[0]
            parsed = self._extract_json_object(response.content or "")
            keywords = parsed.get("keywords", [])
            tags = parsed.get("tags", [])
            return {
                "keywords": [str(item) for item in keywords][:8],
                "context": str(parsed.get("context", "General"))[:300],
                "tags": [str(item) for item in tags][:6],
            }
        except Exception:
            return self._fallback_analysis(content)

    def _normalize_information(self, information: Any) -> str:
        if isinstance(information, str):
            return information
        try:
            return json.dumps(information, ensure_ascii=False, default=str)
        except Exception:
            return str(information)

    def _build_embedding_text(self, note: AMemNote) -> str:
        return (
            f"content: {note.content}\n"
            f"context: {note.context}\n"
            f"keywords: {', '.join(note.keywords)}\n"
            f"tags: {', '.join(note.tags)}"
        )

    def _truncate_for_embedding(self, text: str) -> str:
        if len(text) <= self.MAX_EMBEDDING_CHARS:
            return text
        marker = self.EMBEDDING_TRUNCATION_MARKER
        remaining = self.MAX_EMBEDDING_CHARS - len(marker)
        if remaining <= 0:
            return text[: self.MAX_EMBEDDING_CHARS]
        head = remaining // 2
        tail = remaining - head
        return text[:head] + marker + text[-tail:]

    def _embed_text(self, text: str) -> np.ndarray:
        embedding = text_embedding(
            model=self.embedding_model,
            input=self._truncate_for_embedding(text),
        )
        return np.array(embedding)

    def _link_related_notes(
        self, note_id: str, note_embedding: np.ndarray, max_links: int = 3
    ) -> List[str]:
        if not self.storage:
            return []

        scored_links: List[tuple[float, Dict[str, Any]]] = []
        for stored in self.storage:
            similarity = cosine_similarity(
                stored["embedding"].reshape(1, -1),
                note_embedding.reshape(1, -1),
            )[0][0]
            if similarity >= self.link_threshold:
                scored_links.append((float(similarity), stored))

        scored_links.sort(key=lambda item: item[0], reverse=True)
        linked_ids: List[str] = []
        for _, stored in scored_links[:max_links]:
            linked_ids.append(str(stored["note"]["note_id"]))
            links = stored["note"].setdefault("links", [])
            if note_id not in links:
                links.append(note_id)
        return linked_ids

    def update(self, key: str, information: Any) -> None:
        content = self._normalize_information(information)
        analysis = self._analyze_content(content)
        note = AMemNote(
            note_id=str(uuid.uuid4()),
            content=content,
            keywords=analysis.get("keywords", []),
            context=analysis.get("context", "General"),
            tags=analysis.get("tags", []),
            source=key,
        )
        embedding = self._embed_text(self._build_embedding_text(note))
        note.links = self._link_related_notes(note.note_id, embedding)
        self.storage.append({"note": asdict(note), "embedding": embedding})

    def retrieve_latest(self) -> Optional[Dict[str, Any]]:
        if not self.storage:
            return None
        return self.storage[-1]["note"]

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.storage:
            return []

        query_embedding = self._embed_text(query)
        scored_notes: List[tuple[float, Dict[str, Any]]] = []
        for stored in self.storage:
            similarity = cosine_similarity(
                stored["embedding"].reshape(1, -1),
                query_embedding.reshape(1, -1),
            )[0][0]
            note = dict(stored["note"])
            note["score"] = float(similarity)
            scored_notes.append((float(similarity), note))

        scored_notes.sort(key=lambda item: item[0], reverse=True)
        limit = top_k or self.retrieval_top_k
        return [note for _, note in scored_notes[:limit]]

    def retrieve_all(self) -> List[Dict[str, Any]]:
        return [dict(item["note"]) for item in self.storage]

    def get_memory_str(self) -> str:
        if not self.storage:
            return ""

        query = self._current_task or self._current_agent_profile
        if query:
            notes = self.retrieve(query, top_k=self.max_memory_context)
        else:
            notes = self.retrieve_all()[-self.max_memory_context :]

        if not notes:
            return ""

        parts = ["[Agentic Memory Baseline]"]
        for note in notes:
            parts.append(f"- Memory ID: {note['note_id']}")
            parts.append(f"  Context: {note.get('context', 'General')}")
            parts.append(f"  Content: {note.get('content', '')[:500]}")
            if note.get("keywords"):
                parts.append(f"  Keywords: {', '.join(note['keywords'][:6])}")
            if note.get("tags"):
                parts.append(f"  Tags: {', '.join(note['tags'][:6])}")
            if note.get("links"):
                parts.append(f"  Related: {', '.join(note['links'][:3])}")
        return "\n".join(parts)

    def __repr__(self) -> str:
        return (
            f"AMemMemory(agent_id={self.agent_id!r}, "
            f"notes={len(self.storage)}, llm_model={self.llm_model!r})"
        )
