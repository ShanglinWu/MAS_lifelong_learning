import json
import os
import threading

import boto3
from beartype import beartype
from beartype.typing import Any, Dict, List, Optional
from litellm.types.utils import Message

from marble.llms.error_handler import api_calling_error_exponential_backoff

# ---------------------------------------------------------------------------
# Singleton Bedrock client (lazy, thread-safe)
# ---------------------------------------------------------------------------
_bedrock_client = None
_bedrock_lock = threading.Lock()

MODEL_ALIAS_MAP: Dict[str, str] = {
    "gpt-3.5-turbo": "google.gemma-3-4b-it",
    "gpt-4": "google.gemma-3-4b-it",
    "gpt-4o": "google.gemma-3-4b-it",
    "gpt-4o-mini": "google.gemma-3-4b-it",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": "google.gemma-3-4b-it",
}

DEFAULT_MODEL = "google.gemma-3-4b-it"


def _get_bedrock_client():
    """Return a lazily-initialized boto3 bedrock-runtime client."""
    global _bedrock_client
    if _bedrock_client is None:
        with _bedrock_lock:
            if _bedrock_client is None:
                region = os.environ.get("AWS_REGION", "us-west-2")
                _bedrock_client = boto3.client(
                    "bedrock-runtime", region_name=region
                )
    return _bedrock_client


def _resolve_model(llm_model: str) -> str:
    """Map legacy model names to Bedrock model IDs."""
    # Strip common prefixes added by litellm / together routing
    cleaned = llm_model
    for prefix in ("together_ai/", "together_ai/TA/"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    return MODEL_ALIAS_MAP.get(cleaned, cleaned)


# ---------------------------------------------------------------------------
# Format converters: OpenAI ↔ Bedrock Converse API
# ---------------------------------------------------------------------------

def _convert_messages_for_bedrock(messages: List[Dict[str, str]]):
    """
    Split an OpenAI-style message list into the Bedrock Converse format:
    - system prompts go into a separate list
    - user / assistant messages become the conversation list
    - Bedrock requires strict user/assistant alternation;
      consecutive same-role messages are merged.
    """
    system_prompts: List[Dict[str, Any]] = []
    conversation: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system_prompts.append({"text": content})
            continue

        # Bedrock only knows "user" and "assistant"
        bedrock_role = "assistant" if role == "assistant" else "user"
        bedrock_msg = {
            "role": bedrock_role,
            "content": [{"text": content}],
        }

        # Merge consecutive messages with the same role
        if conversation and conversation[-1]["role"] == bedrock_role:
            conversation[-1]["content"].append({"text": content})
        else:
            conversation.append(bedrock_msg)

    # Bedrock requires the conversation to start with a user message
    if conversation and conversation[0]["role"] != "user":
        conversation.insert(
            0, {"role": "user", "content": [{"text": "Hello."}]}
        )

    # Ensure alternation by inserting filler messages where needed
    fixed: List[Dict[str, Any]] = []
    for msg in conversation:
        if fixed and fixed[-1]["role"] == msg["role"]:
            filler_role = "assistant" if msg["role"] == "user" else "user"
            fixed.append(
                {"role": filler_role, "content": [{"text": "..."}]}
            )
        fixed.append(msg)

    return system_prompts, fixed


def _convert_tools_for_bedrock(
    tools: Optional[List[Dict[str, Any]]],
    tool_choice: Optional[str] = None,
) -> tuple:
    """
    Convert OpenAI-style tool definitions to Bedrock toolConfig format.
    Returns (tool_config_dict_or_None, tool_choice_dict_or_None).
    """
    if not tools:
        return None, None

    bedrock_tools = []
    for tool_def in tools:
        func = tool_def.get("function", {})
        params = func.get("parameters", {})
        # Bedrock requires the inputSchema to have the json wrapper
        spec = {
            "toolSpec": {
                "name": func.get("name", "unknown"),
                "description": func.get("description", ""),
                "inputSchema": {"json": params},
            }
        }
        bedrock_tools.append(spec)

    tool_config: Dict[str, Any] = {"tools": bedrock_tools}

    if tool_choice == "required":
        tool_config["toolChoice"] = {"any": {}}
    elif tool_choice == "auto":
        tool_config["toolChoice"] = {"auto": {}}

    return tool_config, None


def _parse_bedrock_response(response: Dict[str, Any]) -> Message:
    """
    Parse a Bedrock Converse API response into a litellm Message object
    for backward compatibility with all existing callers.
    """
    output = response.get("output", {})
    message = output.get("message", {})
    content_blocks = message.get("content", [])

    text_parts: List[str] = []
    tool_calls: List[Any] = []

    for block in content_blocks:
        if "text" in block:
            text_parts.append(block["text"])
        elif "toolUse" in block:
            tu = block["toolUse"]
            # Build an object that mimics OpenAI tool_call structure
            tool_call = _ToolCall(
                id=tu.get("toolUseId", ""),
                type="function",
                function=_FunctionCall(
                    name=tu.get("name", ""),
                    arguments=json.dumps(tu.get("input", {})),
                ),
            )
            tool_calls.append(tool_call)

    content_str = "\n".join(text_parts) if text_parts else None

    msg = Message(
        content=content_str,
        role="assistant",
    )
    if tool_calls:
        msg.tool_calls = tool_calls

    return msg


def _estimate_text_tokens(text: str) -> int:
    """
    Rough token estimate for budget planning.

    Uses a conservative 4 chars/token heuristic.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _estimate_message_tokens(
    system_prompts: List[Dict[str, Any]], conversation: List[Dict[str, Any]]
) -> int:
    total_chars = 0
    for sys_msg in system_prompts:
        total_chars += len(sys_msg.get("text", ""))
    for msg in conversation:
        for block in msg.get("content", []):
            total_chars += len(block.get("text", ""))
    return max(1, (total_chars + 3) // 4) if total_chars > 0 else 0


class _FunctionCall:
    """Mimics openai.types.chat.ChatCompletionMessageToolCall.Function"""

    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    """Mimics openai.types.chat.ChatCompletionMessageToolCall"""

    def __init__(self, id: str, type: str, function: _FunctionCall):
        self.id = id
        self.type = type
        self.function = function


# ---------------------------------------------------------------------------
# Main prompting function  (signature unchanged)
# ---------------------------------------------------------------------------

@beartype
@api_calling_error_exponential_backoff(retries=5, base_wait_time=1)
def model_prompting(
    llm_model: str,
    messages: List[Dict[str, str]],
    return_num: Optional[int] = 1,
    max_token_num: Optional[int] = 512,
    temperature: Optional[float] = 0.5,
    top_p: Optional[float] = 0.9,
    stream: Optional[bool] = None,
    mode: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
) -> List[Message]:
    """
    Call Amazon Bedrock Converse API.

    Preserves the exact function signature so all callers
    (base_agent, engine_planner, short_term_memory, long_term_memory)
    work unchanged.
    """
    client = _get_bedrock_client()
    model_id = _resolve_model(llm_model)

    system_prompts, conversation = _convert_messages_for_bedrock(messages)

    # Ensure we have at least one message
    if not conversation:
        conversation = [{"role": "user", "content": [{"text": "Hello."}]}]

    # Build kwargs for converse()
    kwargs: Dict[str, Any] = {
        "modelId": model_id,
        "messages": conversation,
    }

    if system_prompts:
        kwargs["system"] = system_prompts

    # Inference config
    inference_config: Dict[str, Any] = {}
    if max_token_num is not None:
        inference_config["maxTokens"] = max_token_num
    if temperature is not None:
        # Bedrock clamps temperature; 0.0 is valid
        inference_config["temperature"] = temperature
    
    # if top_p is not None:
    #     inference_config["topP"] = top_p
    if inference_config:
        kwargs["inferenceConfig"] = inference_config

    # Tools
    tool_config, _ = _convert_tools_for_bedrock(tools, tool_choice)
    if tool_config:
        kwargs["toolConfig"] = tool_config

    response = client.converse(**kwargs)
    msg = _parse_bedrock_response(response)

    # Budget instrumentation:
    # Prefer Bedrock-reported usage when available, fallback to rough estimates.
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    input_tokens = usage.get("inputTokens")
    output_tokens = usage.get("outputTokens")
    total_tokens = usage.get("totalTokens")

    if input_tokens is None:
        input_tokens = _estimate_message_tokens(system_prompts, conversation)
    if output_tokens is None:
        output_tokens = _estimate_text_tokens(msg.content or "")
    if total_tokens is None:
        total_tokens = int(input_tokens) + int(output_tokens)

    usage_payload = {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
        "model_id": model_id,
        "estimated": not bool(usage),
    }
    try:
        setattr(msg, "token_usage_estimate", usage_payload)
    except Exception:
        # Some message classes may disallow dynamic attributes.
        pass

    return [msg]
