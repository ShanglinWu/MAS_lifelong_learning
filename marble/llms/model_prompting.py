import os
import threading

import litellm
from beartype import beartype
from beartype.typing import Any, Dict, List, Optional
from litellm.types.utils import Message


from marble.llms.error_handler import api_calling_error_exponential_backoff

# ---------------------------------------------------------------------------
# Global token usage counter (thread-safe)
# ---------------------------------------------------------------------------
_token_lock = threading.Lock()
_token_counter: Dict[str, int] = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}


def get_token_usage() -> Dict[str, int]:
    """Return a copy of the accumulated token usage across all model_prompting calls."""
    with _token_lock:
        return dict(_token_counter)


def reset_token_usage() -> None:
    """Reset the accumulated token usage counters to zero."""
    with _token_lock:
        _token_counter["prompt_tokens"] = 0
        _token_counter["completion_tokens"] = 0
        _token_counter["total_tokens"] = 0


@beartype
@api_calling_error_exponential_backoff(retries=5, base_wait_time=1)
def model_prompting(
    llm_model: str,
    messages: List[Dict[str, str]],
    return_num: Optional[int] = 1,
    max_token_num: Optional[int] = 512,
    temperature: Optional[float] = 0.0,
    top_p: Optional[float] = None,
    stream: Optional[bool] = None,
    mode: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Message]:
    """
    Select model via router in LiteLLM with support for function calling.
    """
    # Bedrock does not support some OpenAI params (e.g. tool_choice='auto'); drop them silently
    litellm.drop_params = True

    # Fall back to AWS_ACCESS_KEY_ID env var when no explicit key is provided


    if "together_ai/TA" in llm_model:
        base_url = "https://api.ohmygpt.com/v1"
    else:
        base_url = None

    # Bedrock rejects requests where both temperature and top_p are set.
    # Drop top_p for bedrock models when both are specified.
    if "bedrock" in llm_model and temperature is not None and top_p is not None:
        top_p = None

    completion = litellm.completion(
        model=llm_model,
        messages=messages,
        max_tokens=max_token_num,
        n=return_num,
        top_p=top_p,
        temperature=temperature,
        stream=stream,
        tools=tools,
        tool_choice=tool_choice,
        base_url=base_url,
        # api_key=api_key,
    )
    message_0: Message = completion.choices[0].message
    print(f"Received response from model: {message_0}")

    # Accumulate token usage
    if completion.usage is not None:
        with _token_lock:
            _token_counter["prompt_tokens"] += getattr(completion.usage, "prompt_tokens", 0) or 0
            _token_counter["completion_tokens"] += getattr(completion.usage, "completion_tokens", 0) or 0
            _token_counter["total_tokens"] += getattr(completion.usage, "total_tokens", 0) or 0

    assert message_0 is not None
    assert isinstance(message_0, Message)
    return [message_0]
