import math
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

INF = float(math.inf)
T = TypeVar("T", bound=Callable[..., Any])


def api_calling_error_exponential_backoff(
    retries: int = 5,
    base_wait_time: int = 1,
) -> Callable[[T], T]:
    """Retry a function call with exponential backoff."""

    def decorator(func: T) -> T:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Optional[Any]:
            error_handler_mode = kwargs.get("mode")
            max_retries = 1 if error_handler_mode == "TEST" else retries

            attempts = 0
            while attempts < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception:
                    wait_time = base_wait_time * (2**attempts)
                    time.sleep(wait_time)
                    attempts += 1
            return None

        return wrapper  # type: ignore[return-value]

    return decorator
