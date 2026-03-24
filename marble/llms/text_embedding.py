from typing import Any, List

try:
    import litellm
except ImportError:  # pragma: no cover - optional dependency path
    litellm = None

try:
    from beartype import beartype
except ImportError:  # pragma: no cover - optional dependency path
    def beartype(func: Any) -> Any:
        return func

from marble.llms.error_handler import api_calling_error_exponential_backoff


@beartype
@api_calling_error_exponential_backoff(retries=5, base_wait_time=1)
def text_embedding(model: str, input: str) -> List[float]:
    """Generate a single embedding vector through LiteLLM."""
    if litellm is None:
        raise RuntimeError(
            "litellm is required for text embeddings. Install it before using real embedding calls."
        )

    embedding = litellm.embedding(model=model, input=[input])
    embedding_vector = embedding.data[0]["embedding"]
    assert embedding_vector is not None
    assert isinstance(embedding_vector, list)
    return embedding_vector
