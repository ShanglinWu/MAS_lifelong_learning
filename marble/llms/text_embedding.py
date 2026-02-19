import json

import boto3
from beartype import beartype
from beartype.typing import List

from .error_handler import api_calling_error_exponential_backoff
from .model_prompting import _get_bedrock_client

DEFAULT_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"

EMBEDDING_ALIAS_MAP = {
    "text-embedding-ada-002": DEFAULT_EMBEDDING_MODEL,
    "text-embedding-3-small": DEFAULT_EMBEDDING_MODEL,
}


def _resolve_embedding_model(model: str) -> str:
    """Map legacy embedding model names to Bedrock model IDs."""
    return EMBEDDING_ALIAS_MAP.get(model, model)


@beartype
@api_calling_error_exponential_backoff(retries=5, base_wait_time=1)
def text_embedding(
    model: str,
    input: str,
) -> List[float]:
    """
    Generate text embeddings using Amazon Bedrock Titan Embeddings.

    Uses invoke_model with the Titan embedding model.
    Keeps the same function signature for backward compatibility.
    """
    client = _get_bedrock_client()
    model_id = _resolve_embedding_model(model)

    body = json.dumps({"inputText": input})

    response = client.invoke_model(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json",
    )

    response_body = json.loads(response["body"].read())
    embedding = response_body.get("embedding", [])

    assert isinstance(embedding, list)
    return embedding
