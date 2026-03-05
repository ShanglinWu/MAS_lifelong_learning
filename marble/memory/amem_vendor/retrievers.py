"""Vendored and adapted Chroma retriever utilities from A-MEM."""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


_EMPTY_RESULTS = {
    "ids": [[]],
    "documents": [[]],
    "metadatas": [[]],
    "distances": [[]],
}


class ChromaRetriever:
    """Vector database retrieval using ChromaDB."""

    def __init__(
        self,
        collection_name: str = "memories",
        model_name: str = "all-MiniLM-L6-v2",
        reset_collection: bool = False,
    ):
        """Initialize ChromaDB retriever.

        Args:
            collection_name: Name of the ChromaDB collection.
            model_name: SentenceTransformer model used by Chroma embeddings.
            reset_collection: If true, remove existing collection before creating.
        """
        self.collection_name = collection_name
        self.client = chromadb.Client()
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_name)

        if reset_collection:
            try:
                self.client.delete_collection(name=collection_name)
            except Exception:
                # Collection may not exist yet.
                pass

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )

    def add_document(self, document: str, metadata: Dict[str, Any], doc_id: str) -> None:
        """Add or update a document in ChromaDB."""
        processed_metadata: Dict[str, str] = {}
        for key, value in metadata.items():
            if isinstance(value, (list, dict)):
                processed_metadata[key] = json.dumps(value)
            else:
                processed_metadata[key] = str(value)

        self.collection.upsert(
            documents=[document],
            metadatas=[processed_metadata],
            ids=[doc_id],
        )

    def delete_document(self, doc_id: str) -> None:
        """Delete a document from ChromaDB by id."""
        self.collection.delete(ids=[doc_id])

    def search(self, query: str, k: int = 5) -> Dict[str, List[List[Any]]]:
        """Search for similar documents."""
        if k <= 0:
            return dict(_EMPTY_RESULTS)
        try:
            results = self.collection.query(query_texts=[query], n_results=max(1, k))
        except Exception:
            return dict(_EMPTY_RESULTS)

        metadatas = results.get("metadatas")
        if metadatas:
            results["metadatas"] = self._convert_metadata_types(metadatas)
        return results

    def _convert_metadata_types(
        self,
        metadatas: List[List[Dict[str, Any]]],
    ) -> List[List[Dict[str, Any]]]:
        """Convert string metadata values back to basic Python types."""
        for query_metadatas in metadatas:
            if isinstance(query_metadatas, list):
                for metadata_dict in query_metadatas:
                    if isinstance(metadata_dict, dict):
                        self._convert_metadata_dict(metadata_dict)
        return metadatas

    def _convert_metadata_dict(self, metadata: Dict[str, Any]) -> None:
        for key, value in metadata.items():
            if not isinstance(value, str):
                continue
            try:
                metadata[key] = ast.literal_eval(value)
            except Exception:
                # Keep plain strings as-is.
                pass
