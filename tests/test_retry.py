# tests/test_retry.py
"""
Test retry utilities
"""

import pytest
import time
from unittest.mock import Mock, patch
from utils.retry import retry_with_backoff, retry_with_backoff_async, is_retryable_error
from exceptions.base import ModelServiceError

class TestRetryLogic:
    """Test retry logic functionality"""
    
    def test_is_retryable_error(self):
        """Test retryable error detection"""
        # Non-retryable errors
        assert not is_retryable_error(Exception("generic error"))
        assert not is_retryable_error(ValueError("bad request"))
        
        # Retryable errors
        retryable_errors = [
            ModelServiceError(code="429", message="Rate limited"),
            ModelServiceError(code="500", message="Server error"),
            ModelServiceError(code="502", message="Bad gateway"),
            ModelServiceError(code="503", message="Service unavailable"),
            ModelServiceError(code="504", message="Gateway timeout")
        ]
        
        for error in retryable_errors:
            assert is_retryable_error(error)
    
    def test_retry_with_backoff_success(self):
        """Test successful retry after a few attempts"""
        call_count = 0
        
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ModelServiceError(code="500", message="Server error")
            return "success"
        
        result = retry_with_backoff(failing_function, max_retries=5, initial_delay=0.01)
        
        assert result == "success"
        assert call_count == 3
    
    def test_retry_with_backoff_max_retries_exceeded(self):
        """Test retry failure after max retries"""
        call_count = 0

        def always_failing():
            nonlocal call_count
            call_count += 1
            raise ModelServiceError(code="500", message="Server error")

        with pytest.raises(ModelServiceError) as exc_info:
            retry_with_backoff(always_failing, max_retries=3, initial_delay=0.01)

        # The function should be called 1 (initial) + 3 (retries) = 4 times
        assert call_count == 4  # Update expected count
    
    def test_retry_with_backoff_non_retryable_error(self):
        """Test that non-retryable errors are not retried"""
        call_count = 0
        
        def non_retryable_error():
            nonlocal call_count
            call_count += 1
            raise ModelServiceError(code="400", message="Bad request")
        
        with pytest.raises(ModelServiceError) as exc_info:
            retry_with_backoff(non_retryable_error, max_retries=5)
        
        # Should fail immediately without retry
        assert call_count == 1
        assert exc_info.value.code == "400"

@pytest.mark.asyncio
class TestAsyncRetry:
    """Test async retry functionality"""
    
    async def test_async_retry_with_backoff_success(self):
        """Test successful async retry"""
        call_count = 0
        
        async def async_failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ModelServiceError(code="500", message="Server error")
            return "success"
        
        result = await retry_with_backoff_async(async_failing_function, max_retries=5, initial_delay=0.01)
        
        assert result == "success"
        assert call_count == 3
    
    async def test_async_retry_max_retries_exceeded(self):
        """Test async retry failure after max retries"""
        call_count = 0
    
        async def async_always_failing():
            nonlocal call_count
            call_count += 1
            raise ModelServiceError(code="500", message="Server error")
    
        with pytest.raises(ModelServiceError):
            await retry_with_backoff_async(async_always_failing, max_retries=3, initial_delay=0.01)
    
        # The function should be called 1 (initial) + 3 (retries) = 4 times
        assert call_count == 4  # Update expected count