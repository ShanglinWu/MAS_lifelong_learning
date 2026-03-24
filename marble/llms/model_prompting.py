import threading
from typing import Any, Dict, List, Optional

try:
    import litellm
    from litellm.types.utils import Message
except ImportError:  # pragma: no cover - optional dependency path
    litellm = None
    Message = Any

try:
    from beartype import beartype
except ImportError:  # pragma: no cover - optional dependency path
    def beartype(func: Any) -> Any:
        return func

from marble.llms.error_handler import api_calling_error_exponential_backoff

_token_lock = threading.Lock()
_token_counter: Dict[str, int] = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}


def get_token_usage() -> Dict[str, int]:
    """Return aggregated token usage across all model calls."""
    with _token_lock:
        return dict(_token_counter)


def reset_token_usage() -> None:
    """Reset aggregated token usage counters."""
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
    max_token_num: Optional[int] = 1024,
    temperature: Optional[float] = 0.0,
    top_p: Optional[float] = None,
    stream: Optional[bool] = None,
    mode: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Message]:
    """Send a completion request through LiteLLM."""
    del max_token_num, mode, api_key

    if litellm is None:
        raise RuntimeError(
            "litellm is required for model prompting. Install it before using real LLM calls."
        )

    litellm.drop_params = True

    base_url = "https://api.ohmygpt.com/v1" if "together_ai/TA" in llm_model else None
    if "bedrock" in llm_model and temperature is not None and top_p is not None:
        top_p = None

    completion = litellm.completion(
        model=llm_model,
        messages=messages,
        n=return_num,
        top_p=top_p,
        temperature=temperature,
        stream=stream,
        tools=tools,
        tool_choice=tool_choice,
        base_url=base_url,
    )
    message = completion.choices[0].message

    if completion.usage is not None:
        with _token_lock:
            _token_counter["prompt_tokens"] += (
                getattr(completion.usage, "prompt_tokens", 0) or 0
            )
            _token_counter["completion_tokens"] += (
                getattr(completion.usage, "completion_tokens", 0) or 0
            )
            _token_counter["total_tokens"] += (
                getattr(completion.usage, "total_tokens", 0) or 0
            )

    return [message]
