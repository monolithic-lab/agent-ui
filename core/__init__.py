# core/__init__.py
"""
Core module for agent-ui framework
Contains essential infrastructure components
"""

from .exceptions import (
    ModelServiceError,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    ValidationError,
    ToolExecutionError,
    LoopDetectionError
)

from .schemas import (
    Message,
    ASSISTANT,
    USER,
    SYSTEM,
    TOOL_RESULT,
    FunctionCall,
    ToolCall,
    CHAT_COMPLETION,
    CHAT_MESSAGE
)

from .database import (
    DatabaseManager,
    Session,
    UserSettings,
    Message as DatabaseMessage
)

from .provider import (
    BaseProvider,
    ModelResponse,
    ProviderInfo,
    ProviderFactory,
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    HuggingFaceProvider
)

from .client import (
    MCPClientSystem,
    ConversationManager,
    WebSocketManager
)

__all__ = [
    # Exceptions
    'ModelServiceError',
    'ProviderError', 
    'RateLimitError',
    'AuthenticationError',
    'ValidationError',
    'ToolExecutionError',
    'LoopDetectionError',
    
    # Schemas
    'Message',
    'ASSISTANT',
    'USER',
    'SYSTEM',
    'TOOL_RESULT',
    'FunctionCall',
    'ToolCall',
    'CHAT_COMPLETION',
    'CHAT_MESSAGE',
    
    # Database
    'DatabaseManager',
    'Session',
    'UserSettings',
    'DatabaseMessage',
    
    # Provider
    'BaseProvider',
    'ModelResponse',
    'ProviderInfo',
    'ProviderFactory',
    'OpenAIProvider',
    'AnthropicProvider',
    'GeminiProvider',
    'HuggingFaceProvider',
    
    # Client
    'MCPClientSystem',
    'ConversationManager',
    'WebSocketManager'
]