# utils/retry.py
import random
import time
from typing import Any, Callable, Tuple, Optional
from exceptions.base import ModelServiceError

def retry_with_backoff(
    fn: Callable,
    max_retries: int = 10,
    initial_delay: float = 1.0,
    max_delay: float = 300.0,
    exponential_base: float = 2.0,
    jitter: float = 0.1,
) -> Any:
    """Retry with exponential backoff"""
    
    num_retries, delay = 0, initial_delay
    while True:
        try:
            return fn()
        except ModelServiceError as e:
            num_retries, delay = _handle_retry_error(e, num_retries, delay, max_retries, max_delay, exponential_base, jitter)

def retry_with_backoff_async(
    coro_fn: Callable,
    max_retries: int = 10,
    initial_delay: float = 1.0,
    max_delay: float = 300.0,
    exponential_base: float = 2.0,
    jitter: float = 0.1,
) -> Any:
    """Async retry with exponential backoff"""
    
    async def wrapper():
        num_retries, delay = 0, initial_delay
        while True:
            try:
                return await coro_fn()
            except ModelServiceError as e:
                num_retries, delay = _handle_retry_error(e, num_retries, delay, max_retries, max_delay, exponential_base, jitter)
    
    return wrapper()

def _handle_retry_error(
    e: ModelServiceError,
    num_retries: int,
    delay: float,
    max_retries: int,
    max_delay: float,
    exponential_base: float,
    jitter: float,
) -> Tuple[int, float]:
    """Handle retry logic"""
    
    # Don't retry bad requests
    if e.code == '400':
        raise e
    
    # Don't retry content filtering
    if e.code == 'DataInspectionFailed':
        raise e
    if 'inappropriate content' in str(e):
        raise e
    
    # Don't retry context length issues
    if 'maximum context length' in str(e):
        raise e
    
    if num_retries >= max_retries:
        raise ModelServiceError(
            exception=Exception(f'Maximum number of retries ({max_retries}) exceeded.')
        )
    
    num_retries += 1
    # Add jitter to prevent thundering herd
    jitter_amount = jitter * random.random()
    delay = min(delay * exponential_base, max_delay) * (1 + jitter_amount)
    time.sleep(delay)
    return num_retries, delay

def is_retryable_error(error: Exception) -> bool:
    """Check if error is retryable"""
    if isinstance(error, ModelServiceError):
        # Retry rate limits and server errors
        return error.code in ['429', '500', '502', '503', '504']
    return False