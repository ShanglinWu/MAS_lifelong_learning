"""Vendored and adapted agentic memory system from A-MEM."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .llm_controller import LLMController
from .retrievers import ChromaRetriever

logger = logging.getLogger(__name__)


class MemoryNote:
    """A single note in the A-MEM store."""

    def __init__(
        self,
        content: str,
        id: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        links: Optional[List[str]] = None,
        retrieval_count: Optional[int] = None,
        timestamp: Optional[str] = None,
        last_accessed: Optional[str] = None,
        context: Optional[str] = None,
        evolution_history: Optional[List[str]] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        self.content = content
        self.id = id or str(uuid.uuid4())

        self.keywords = keywords or []
        self.links = links or []
        self.context = context or "General"
        self.category = category or "Uncategorized"
        self.tags = tags or []

        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time

        self.retrieval_count = retrieval_count or 0
        self.evolution_history = evolution_history or []


class AgenticMemorySystem:
    """Core memory system that manages memory notes and their evolution."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        llm_backend: str = "openai",
        llm_model: str = "gpt-4o-mini",
        evo_threshold: int = 100,
        api_key: Optional[str] = None,
        collection_name: str = "memories",
        reset_collection: bool = False,
        evolution_enabled: bool = True,
    ):
        self.memories: Dict[str, MemoryNote] = {}
        self.model_name = model_name
        self.collection_name = collection_name
        self.evo_threshold = max(1, int(evo_threshold))
        self.evo_cnt = 0
        self.evolution_enabled = bool(evolution_enabled)

        # No global reset side effects. Reset behavior is explicit and scoped.
        self.retriever = ChromaRetriever(
            collection_name=self.collection_name,
            model_name=self.model_name,
            reset_collection=reset_collection,
        )

        self.llm_controller: Optional[LLMController] = None
        if self.evolution_enabled:
            self.llm_controller = LLMController(llm_backend, llm_model, api_key)

        self._evolution_system_prompt = """
You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
Analyze the new memory note according to keywords and context, with nearest neighbor memories.

The new memory context:
{context}
content: {content}
keywords: {keywords}

The nearest neighbors memories:
{nearest_neighbors_memories}

Based on this information, determine:
1. Should this memory be evolved?
2. What actions should be taken (strengthen, update_neighbor)?
3. If strengthen, suggest linked memory ids and updated tags.
4. If update_neighbor, return context/tags updates aligned to the same neighbor order.

Return JSON with this shape:
{
  "should_evolve": true/false,
  "actions": ["strengthen", "update_neighbor"],
  "suggested_connections": ["neighbor_memory_ids"],
  "tags_to_update": ["tag_1", "tag_2"],
  "new_context_neighborhood": ["context_a", "context_b"],
  "new_tags_neighborhood": [["tag_a"], ["tag_b"]]
}
""".strip()

    def _note_metadata(self, note: MemoryNote) -> Dict[str, Any]:
        return {
            "id": note.id,
            "content": note.content,
            "keywords": note.keywords,
            "links": note.links,
            "retrieval_count": note.retrieval_count,
            "timestamp": note.timestamp,
            "last_accessed": note.last_accessed,
            "context": note.context,
            "evolution_history": note.evolution_history,
            "category": note.category,
            "tags": note.tags,
        }

    def add_note(self, content: str, time: Optional[str] = None, **kwargs: Any) -> str:
        """Add a new memory note."""
        if time is not None:
            kwargs["timestamp"] = time
        note = MemoryNote(content=content, **kwargs)

        evo_label = False
        if self.evolution_enabled:
            evo_label, note = self.process_memory(note)

        self.memories[note.id] = note
        self.retriever.add_document(note.content, self._note_metadata(note), note.id)

        if evo_label:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id

    def consolidate_memories(self) -> None:
        """Rebuild collection from in-memory notes."""
        self.retriever = ChromaRetriever(
            collection_name=self.collection_name,
            model_name=self.model_name,
            reset_collection=True,
        )
        for memory in self.memories.values():
            self.retriever.add_document(
                memory.content,
                self._note_metadata(memory),
                memory.id,
            )

    def find_related_memories(self, query: str, k: int = 5) -> Tuple[str, List[str]]:
        """Find related memories and return prompt text + neighbor ids."""
        if not self.memories:
            return "", []

        try:
            results = self.retriever.search(query, k)
            memory_str = ""
            neighbor_ids: List[str] = []
            ids = results.get("ids", [[]])
            metadatas = results.get("metadatas", [[]])
            if not ids or not ids[0]:
                return "", []

            for i, doc_id in enumerate(ids[0]):
                if i >= len(metadatas[0]):
                    continue
                metadata = metadatas[0][i]
                memory_str += (
                    f"memory id:{doc_id}\t"
                    f"talk start time:{metadata.get('timestamp', '')}\t"
                    f"memory content: {metadata.get('content', '')}\t"
                    f"memory context: {metadata.get('context', '')}\t"
                    f"memory keywords: {str(metadata.get('keywords', []))}\t"
                    f"memory tags: {str(metadata.get('tags', []))}\n"
                )
                neighbor_ids.append(str(doc_id))
            return memory_str, neighbor_ids
        except Exception as exc:
            logger.error("Error in find_related_memories: %s", exc)
            return "", []

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """Find related memories in a compact text format."""
        text, _ = self.find_related_memories(query, k=k)
        return text

    def read(self, memory_id: str) -> Optional[MemoryNote]:
        return self.memories.get(memory_id)

    def update(self, memory_id: str, **kwargs: Any) -> bool:
        if memory_id not in self.memories:
            return False

        note = self.memories[memory_id]
        for key, value in kwargs.items():
            if hasattr(note, key):
                setattr(note, key, value)

        self.retriever.add_document(
            document=note.content,
            metadata=self._note_metadata(note),
            doc_id=memory_id,
        )
        return True

    def delete(self, memory_id: str) -> bool:
        if memory_id in self.memories:
            self.retriever.delete_document(memory_id)
            del self.memories[memory_id]
            return True
        return False

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Simple search wrapper returning compact result dicts."""
        search_results = self.retriever.search(query, k)
        rows: List[Dict[str, Any]] = []
        ids = search_results.get("ids", [[]])
        distances = search_results.get("distances", [[]])

        for i, doc_id in enumerate(ids[0]):
            memory = self.memories.get(doc_id)
            if memory is None:
                continue
            score = distances[0][i] if i < len(distances[0]) else None
            rows.append(
                {
                    "id": doc_id,
                    "content": memory.content,
                    "context": memory.context,
                    "keywords": memory.keywords,
                    "tags": memory.tags,
                    "links": memory.links,
                    "score": score,
                }
            )
        return rows[:k]

    def search_agentic(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search memories with optional linked-neighbor expansion."""
        if not self.memories:
            return []

        try:
            results = self.retriever.search(query, k)
            memories: List[Dict[str, Any]] = []
            seen_ids = set()

            ids = results.get("ids", [[]])
            metadatas = results.get("metadatas", [[]])
            distances = results.get("distances", [[]])
            if not ids or not ids[0]:
                return []

            for i, doc_id in enumerate(ids[0][:k]):
                if doc_id in seen_ids or i >= len(metadatas[0]):
                    continue
                metadata = metadatas[0][i]
                row = {
                    "id": doc_id,
                    "content": metadata.get("content", ""),
                    "context": metadata.get("context", ""),
                    "keywords": metadata.get("keywords", []),
                    "tags": metadata.get("tags", []),
                    "links": metadata.get("links", []),
                    "timestamp": metadata.get("timestamp", ""),
                    "category": metadata.get("category", "Uncategorized"),
                    "is_neighbor": False,
                }
                if distances and distances[0] and i < len(distances[0]):
                    row["score"] = distances[0][i]
                memories.append(row)
                seen_ids.add(doc_id)

            neighbor_budget = k
            for memory in list(memories):
                if neighbor_budget <= 0:
                    break
                links = memory.get("links", [])
                if not links:
                    mem_obj = self.memories.get(memory.get("id", ""))
                    links = mem_obj.links if mem_obj else []

                for link_id in links:
                    if neighbor_budget <= 0:
                        break
                    if link_id in seen_ids:
                        continue
                    neighbor = self.memories.get(link_id)
                    if neighbor is None:
                        continue
                    memories.append(
                        {
                            "id": link_id,
                            "content": neighbor.content,
                            "context": neighbor.context,
                            "keywords": neighbor.keywords,
                            "tags": neighbor.tags,
                            "links": neighbor.links,
                            "timestamp": neighbor.timestamp,
                            "category": neighbor.category,
                            "is_neighbor": True,
                        }
                    )
                    seen_ids.add(link_id)
                    neighbor_budget -= 1

            return memories[:k]
        except Exception as exc:
            logger.error("Error in search_agentic: %s", exc)
            return []

    def process_memory(self, note: MemoryNote) -> Tuple[bool, MemoryNote]:
        """Run optional LLM-guided memory evolution."""
        if not self.evolution_enabled or self.llm_controller is None:
            return False, note
        if not self.memories:
            return False, note

        neighbors_text, neighbor_ids = self.find_related_memories(note.content, k=5)
        if not neighbors_text or not neighbor_ids:
            return False, note

        prompt = self._evolution_system_prompt.format(
            content=note.content,
            context=note.context,
            keywords=note.keywords,
            nearest_neighbors_memories=neighbors_text,
            neighbor_number=len(neighbor_ids),
        )

        response_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "should_evolve": {"type": "boolean"},
                        "actions": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "suggested_connections": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "new_context_neighborhood": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "tags_to_update": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "new_tags_neighborhood": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                    "required": [
                        "should_evolve",
                        "actions",
                        "suggested_connections",
                        "tags_to_update",
                        "new_context_neighborhood",
                        "new_tags_neighborhood",
                    ],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        try:
            response = self.llm_controller.get_completion(
                prompt,
                response_format=response_schema,
            )
            response_json = json.loads(response)
        except Exception as exc:
            logger.error("Error in memory evolution: %s", exc)
            return False, note

        should_evolve = bool(response_json.get("should_evolve", False))
        if not should_evolve:
            return False, note

        actions = response_json.get("actions", [])
        if "strengthen" in actions:
            suggested_connections = [
                str(x) for x in response_json.get("suggested_connections", []) if x
            ]
            note.links = list(dict.fromkeys(note.links + suggested_connections))
            updated_tags = response_json.get("tags_to_update", [])
            if isinstance(updated_tags, list) and updated_tags:
                note.tags = [str(x) for x in updated_tags]

        if "update_neighbor" in actions:
            new_contexts = response_json.get("new_context_neighborhood", [])
            new_tags = response_json.get("new_tags_neighborhood", [])

            for idx, neighbor_id in enumerate(neighbor_ids):
                neighbor = self.memories.get(neighbor_id)
                if neighbor is None:
                    continue
                if idx < len(new_contexts) and isinstance(new_contexts[idx], str):
                    neighbor.context = new_contexts[idx]
                if idx < len(new_tags) and isinstance(new_tags[idx], list):
                    neighbor.tags = [str(x) for x in new_tags[idx]]
                self.memories[neighbor_id] = neighbor

        return True, note
