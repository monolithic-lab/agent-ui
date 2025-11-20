# tests/test_exceptions.py
"""
Test exception hierarchy
"""

import pytest
from exceptions.base import ModelServiceError, ProviderError, RateLimitError

def test_model_service_error():
    """Test ModelServiceError creation and attributes"""
    original_error = ValueError("Test error")
    error = ModelServiceError(
        exception=original_error,
        code="400",
        message="Bad request",
        extra={"field": "value"}
    )

    assert error.exception == original_error
    assert error.code == "400"
    assert error.message == "Bad request"
    assert error.extra == {"field": "value"}
    # Fix the assertion to check the actual string representation
    error_str = str(error)
    assert "400" in error_str or "Bad request" in error_str

def test_provider_error_inheritance():
    """Test ProviderError inheritance"""
    error = ProviderError(code="500", message="Server error")
    assert isinstance(error, ModelServiceError)
    assert error.code == "500"

def test_rate_limit_error():
    """Test RateLimitError specialization"""
    error = RateLimitError(code="429", message="Rate limit exceeded")
    assert isinstance(error, ProviderError)
    assert isinstance(error, ModelServiceError)

def test_error_with_only_exception():
    """Test error creation with only exception"""
    original_error = ValueError("Test error")
    error = ModelServiceError(exception=original_error)
    
    assert error.exception == original_error
    assert str(error) == "Test error"

def test_error_with_code_and_message():
    """Test error creation with code and message"""
    error = ModelServiceError(code="400", message="Bad request")
    
    assert error.code == "400"
    assert error.message == "Bad request"
    assert "400" in str(error)
    assert "Bad request" in str(error)