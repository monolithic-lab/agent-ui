# core/exceptions.py
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class ModelServiceError(Exception):
    """Structured error with code, message, and extra metadata"""
    
    exception: Optional[Exception] = None
    code: Optional[str] = None
    message: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None
    
    def __init__(self, 
                 exception: Optional[Exception] = None,
                 code: Optional[str] = None,
                 message: Optional[str] = None,
                 extra: Optional[Dict[str, Any]] = None):
        
        if exception is not None:
            super().__init__(exception)
        else:
            super().__init__(f'\nError code: {code}. Error message: {message}')
            
        self.exception = exception
        self.code = code
        self.message = message
        self.extra = extra or {}

class ProviderError(ModelServiceError):
    """Base class for provider-specific errors"""
    pass

class RateLimitError(ProviderError):
    """Rate limit exceeded error"""
    pass

class AuthenticationError(ProviderError):
    """Authentication failed error"""
    pass

class ValidationError(ProviderError):
    """Input validation failed"""
    pass

class ToolExecutionError(ModelServiceError):
    """Tool execution failed error"""
    pass

class LoopDetectionError(ModelServiceError):
    """Infinite loop detected"""
    pass