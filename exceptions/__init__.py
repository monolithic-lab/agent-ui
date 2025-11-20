# exceptions/__init__.py
"""Exception hierarchy for agent-ui framework"""

from .base import (
    ModelServiceError,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    ValidationError,
    ToolExecutionError,
    LoopDetectionError
)

__all__ = [
    'ModelServiceError',
    'ProviderError',
    'RateLimitError',
    'AuthenticationError',
    'ValidationError',
    'ToolExecutionError',
    'LoopDetectionError'
]