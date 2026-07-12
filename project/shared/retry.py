"""
Retry decorator with exponential backoff.

Usage:
    from deeputin.shared.retry import retry
    
    @retry(max_attempts=3, backoff_base=1.0)
    def load_data(path):
        # If this fails, retries after 1s, 2s, 4s
        return open(path).read()
"""

from __future__ import annotations

import functools
import time
from typing import Callable, TypeVar, Any

from .logging import setup_logger

log = setup_logger("retry")

F = TypeVar("F", bound=Callable[..., Any])


def retry(
    max_attempts: int = 3,
    backoff_base: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[F], F]:
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts (default: 3)
        backoff_base: Base delay in seconds (default: 1.0)
        backoff_factor: Multiplier for each retry (default: 2.0)
        exceptions: Tuple of exception types to catch (default: all)
        on_retry: Optional callback(exception, attempt_number)
    
    Example:
        @retry(max_attempts=3, backoff_base=1.0)
        def load_json(path):
            with open(path) as f:
                return json.load(f)
        
        # Attempts: 1st try, wait 1s, 2nd try, wait 2s, 3rd try, fail
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        log.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    
                    delay = backoff_base * (backoff_factor ** (attempt - 1))
                    log.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    
                    if on_retry:
                        on_retry(e, attempt)
                    
                    time.sleep(delay)
            
            # Should never reach here, but just in case
            raise last_exception
        
        return wrapper  # type: ignore
    return decorator
