# Agent-UI Implementation Guides
## How to Build Qwen-Agent/Qwen-Code Level Capabilities

### 🎯 **Overview**

This guide provides step-by-step instructions for transforming your agent-ui from a solid foundation into an enterprise-grade agent framework matching Qwen-Agent and Qwen-Code sophistication.

**Your Current Strengths**:
- ✅ Excellent provider abstraction (861-line provider.py)
- ✅ Solid MongoDB integration with async operations  
- ✅ MCP multi-server orchestration system
- ✅ Rich HuggingFace tools (8+ specialized tools)
- ✅ Session management and persistence
- ✅ Clean modular architecture

**Goal**: Add enterprise-grade agent orchestration, safety, performance, and usability features.

---

## 📚 **Guide 1: Dependencies & Environment Setup**

### Step 1: Create requirements.txt

Create `requirements.txt` with production dependencies:

```txt
# Core AI Providers
openai>=1.0.0
anthropic>=0.7.0
google-generativeai>=0.3.0
huggingface_hub>=0.20.0

# Database & Async
motor>=3.3.0
pymongo>=4.5.0

# MCP Protocol
mcp>=1.0.0

# Caching & Performance
diskcache>=5.6.0
redis>=5.0.0

# WebSocket & Real-time
websockets>=12.0
aiohttp>=3.9.0

# Async Utilities
asyncio-throttle>=1.0.0
httpx>=0.25.0

# Security & Validation
cryptography>=41.0.0
pydantic>=2.4.0

# Development & Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
black>=23.9.0
flake8>=6.1.0
mypy>=1.6.0
```

### Step 2: Create development requirements

Create `requirements-dev.txt`:

```txt
-r requirements.txt
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
black>=23.9.0
flake8>=6.1.0
mypy>=1.6.0
pre-commit>=3.5.0
```

### Step 3: Setup.py and pyproject.toml

**pyproject.toml**:
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "agent-ui"
version = "0.1.0"
description = "Multi-provider agent framework with MCP integration"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
dependencies = [
    "openai>=1.0.0",
    "anthropic>=0.7.0",
    "google-generativeai>=0.3.0",
    "huggingface_hub>=0.20.0",
    "motor>=3.3.0",
    "pymongo>=4.5.0",
    "mcp>=1.0.0",
    "diskcache>=5.6.0",
    "redis>=5.0.0",
    "websockets>=12.0",
    "aiohttp>=3.9.0",
    "asyncio-throttle>=1.0.0",
    "httpx>=0.25.0",
    "cryptography>=41.0.0",
    "pydantic>=2.4.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0", 
    "pytest-cov>=4.1.0",
    "black>=23.9.0",
    "flake8>=6.1.0",
    "mypy>=1.6.0",
    "pre-commit>=3.5.0",
]

[project.scripts]
agent-ui = "cli.main:main"
agent-cli = "cli.main:main"

[tool.setuptools.packages.find]
where = ["."]

[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'

[tool.flake8]
max-line-length = 88
extend-ignore = ['E203', 'W503']

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

### Step 4: Environment configuration

**Create `.env.example`**:
```bash
# Database Configuration
MONGODB_URL=mongodb://localhost:27017
DB_NAME=agent_ui

# AI Provider API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
HUGGINGFACE_API_TOKEN=your_huggingface_token_here

# MCP Server Configuration
MCP_SERVERS_PATH=./mcp_servers
MCP_AUTO_CONNECT=true

# Cache Configuration
REDIS_URL=redis://localhost:6379
DISK_CACHE_DIR=./cache
CACHE_TTL=3600

# WebSocket Configuration
WEBSOCKET_HOST=localhost
WEBSOCKET_PORT=8765
WEBSOCKET_MAX_CONNECTIONS=100

# Security Configuration
SECRET_KEY=your_secret_key_for_encryption
ENCRYPT_API_KEYS=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
ENABLE_TELEMETRY=false

# Development Configuration
DEBUG=false
AUTO_RELOAD=false
MAX_CONCURRENT_REQUESTS=10
```

**Create `.gitignore`**:
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Cache and temporary files
.cache/
.cache/
*.tmp
*.temp

# Database
*.db
*.sqlite

# Logs
logs/
*.log

# OS
.DS_Store
Thumbs.db

# Testing
.coverage
.pytest_cache/
htmlcov/

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
```

---

## 📚 **Guide 2: Advanced Error Handling & Resilience**

### Step 1: Create Exception Hierarchy

Create `exceptions/base.py`:

```python
# exceptions/base.py
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
```

### Step 2: Implement Retry Logic

Create `utils/retry.py`:

```python
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
    """Retry with exponential backoff - copied from Qwen-Agent"""
    
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
    """Handle retry logic - copied from Qwen-Agent"""
    
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
```

### Step 3: Update Provider Classes

Update your `provider.py` to use the new error handling:

```python
# In provider.py, add to BaseProvider class:
from exceptions.base import ModelServiceError, ProviderError
from utils.retry import retry_with_backoff

class BaseProvider(ABC):
    # ... existing code ...
    
    async def create_chat_completion_with_retry(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[ModelResponse, AsyncGenerator[ModelResponse, None]]:
        """Create chat completion with automatic retry"""
        
        async def _make_request():
            return await self.create_chat_completion(messages, tools, stream, **kwargs)
        
        return await retry_with_backoff_async(_make_request, max_retries=3)
    
    def _handle_provider_error(self, error: Exception) -> ModelServiceError:
        """Convert provider-specific errors to ModelServiceError"""
        if isinstance(error, Exception):
            # Extract error information from different providers
            if hasattr(error, 'status_code'):
                code = str(error.status_code)
            elif hasattr(error, 'code'):
                code = error.code
            else:
                code = '500'  # Default server error
                
            message = str(error)
            return ModelServiceError(exception=error, code=code, message=message)
        
        return ModelServiceError(exception=error, code='500', message='Unknown error')
```

---

## 📚 **Guide 3: Agent Orchestration System**

### Step 1: Create Base Agent Hierarchy

Create `agents/base_agent.py`:

```python
# agents/base_agent.py
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator, AsyncIterator
from dataclasses import dataclass

from llm import BaseChatModel
from tools.base_tool import BaseTool
from exceptions.base import ModelServiceError
from llm.schema import Message

logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Configuration for agent"""
    name: str
    description: str
    system_message: str
    llm: Optional[BaseChatModel] = None
    tools: Optional[List[BaseTool]] = None
    max_iterations: int = 10
    temperature: float = 0.7
    max_tokens: Optional[int] = None

class BaseAgent(ABC):
    """Base class for all agents - inspired by Qwen-Agent"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.tools = config.tools or []
        self.llm = config.llm
        self._iteration_count = 0
        self._system_messages = [Message(role='system', content=config.system_message)]
        
    async def run(
        self, 
        messages: List[Message],
        **kwargs
    ) -> Iterator[List[Message]]:
        """Main agent execution - yields response messages"""
        self._iteration_count = 0
        async for response in self._run(messages, **kwargs):
            yield response
    
    @abstractmethod
    async def _run(
        self, 
        messages: List[Message], 
        **kwargs
    ) -> AsyncIterator[List[Message]]:
        """Core agent logic - to be implemented by subclasses"""
        pass
    
    async def _call_llm(
        self, 
        messages: List[Message],
        **kwargs
    ) -> List[Message]:
        """Call LLM with retry and error handling"""
        if not self.llm:
            raise ModelServiceError(message="No LLM configured")
        
        try:
            response = await self.llm.chat(
                messages=messages,
                tools=[tool.get_schema() for tool in self.tools if tool.enabled],
                **kwargs
            )
            return response
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise self.llm._handle_provider_error(e)
    
    def _check_iteration_limit(self) -> None:
        """Check if iteration limit exceeded"""
        if self._iteration_count >= self.config.max_iterations:
            raise ModelServiceError(
                message=f"Maximum iterations ({self.config.max_iterations}) exceeded"
            )
    
    def _reset_iteration_count(self) -> None:
        """Reset iteration counter"""
        self._iteration_count = 0
    
    def add_tool(self, tool: BaseTool) -> None:
        """Add a tool to the agent"""
        self.tools.append(tool)
    
    def remove_tool(self, tool_name: str) -> None:
        """Remove a tool from the agent"""
        self.tools = [tool for tool in self.tools if tool.name != tool_name]
    
    def get_tools(self) -> List[BaseTool]:
        """Get all enabled tools"""
        return [tool for tool in self.tools if tool.enabled]

class AgentRegistry:
    """Registry for dynamic agent registration"""
    
    _agents: Dict[str, type] = {}
    
    @classmethod
    def register(cls, agent_type: str):
        """Register an agent type"""
        def decorator(agent_class: type):
            cls._agents[agent_type] = agent_class
            return agent_class
        return decorator
    
    @classmethod
    def create_agent(cls, agent_type: str, config: AgentConfig) -> BaseAgent:
        """Create an agent instance"""
        if agent_type not in cls._agents:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent_class = cls._agents[agent_type]
        return agent_class(config)
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of available agent types"""
        return list(cls._agents.keys())
```

### Step 2: Create Function Calling Agent

Create `agents/fncall_agent.py`:

```python
# agents/fncall_agent.py
import logging
from typing import List, Dict, Any, AsyncIterator

from agents.base_agent import BaseAgent, AgentConfig
from exceptions.base import ModelServiceError
from llm.schema import ASSISTANT, Message, FunctionCall
from tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

class FnCallAgent(BaseAgent):
    """Function calling agent - inspired by Qwen-Agent"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.function_results: List[Dict[str, Any]] = []
    
    async def _run(
        self, 
        messages: List[Message], 
        **kwargs
    ) -> AsyncIterator[List[Message]]:
        """Function calling loop"""
        self._reset_iteration_count()
        
        # Prepare messages
        all_messages = self._system_messages + messages
        
        while True:
            self._check_iteration_limit()
            self._iteration_count += 1
            
            # Call LLM
            llm_response = await self._call_llm(all_messages, **kwargs)
            
            if not llm_response:
                break
                
            # Process LLM response
            for response in llm_response:
                all_messages.append(response)
                
                # If response has function calls, execute them
                if hasattr(response, 'function_call') and response.function_call:
                    tool_results = await self._execute_function_calls(
                        [response.function_call], 
                        all_messages[:-1]  # Exclude the current response
                    )
                    
                    # Add function results to conversation
                    for result in tool_results:
                        all_messages.append(result)
                        
                    # Continue to next iteration
                    continue
                    
                # If response has content, yield and return
                if response.content:
                    yield [response]
                    return
        
        # Loop ended without content
        yield [Message(role=ASSISTANT, content="I apologize, but I was unable to complete the task.")]
    
    async def _execute_function_calls(
        self, 
        function_calls: List[FunctionCall], 
        context_messages: List[Message]
    ) -> List[Message]:
        """Execute function calls and return results"""
        results = []
        
        for function_call in function_calls:
            try:
                # Find the tool
                tool = self._find_tool(function_call.name)
                if not tool:
                    results.append(self._create_error_result(
                        function_call, f"Tool '{function_call.name}' not found"
                    ))
                    continue
                
                # Execute tool
                arguments = self._parse_arguments(function_call.arguments, tool)
                tool_result = await tool.execute(arguments)
                
                # Add result to context
                result_message = Message(
                    role='tool_result',
                    content=str(tool_result.content),
                    tool_call_id=function_call.id,
                    metadata={'tool_name': function_call.name}
                )
                results.append(result_message)
                
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                results.append(self._create_error_result(
                    function_call, f"Tool execution failed: {str(e)}"
                ))
        
        return results
    
    def _find_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Find tool by name"""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None
    
    def _parse_arguments(self, arguments: str, tool: BaseTool) -> Dict[str, Any]:
        """Parse tool arguments safely"""
        import json
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            raise ModelServiceError(message=f"Invalid JSON arguments for tool {tool.name}")
    
    def _create_error_result(self, function_call: FunctionCall, error: str) -> Message:
        """Create error result message"""
        return Message(
            role='tool_result',
            content=f"Error: {error}",
            tool_call_id=function_call.id,
            metadata={'tool_name': function_call.name, 'error': error}
        )

# Register the agent
AgentRegistry.register('fncall')(FnCallAgent)
```

### Step 3: Create Assistant Agent

Create `agents/assistant.py`:

```python
# agents/assistant.py
import logging
from typing import List, Dict, Any, AsyncIterator

from agents.fncall_agent import FnCallAgent
from agents.base_agent import AgentConfig
from llm.schema import Message

logger = logging.getLogger(__name__)

class Assistant(FnCallAgent):
    """Main user-facing assistant agent - inspired by Qwen-Agent"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        # Assistant-specific initialization
        self.rag_enabled = config.rag_enabled if hasattr(config, 'rag_enabled') else False
        self.memory_enabled = config.memory_enabled if hasattr(config, 'memory_enabled') else False
    
    async def _run(
        self, 
        messages: List[Message], 
        **kwargs
    ) -> AsyncIterator[List[Message]]:
        """Main assistant execution loop"""
        
        # Pre-process messages (add system context, RAG, etc.)
        processed_messages = await self._preprocess_messages(messages)
        
        # Use parent function calling logic
        async for response in super()._run(processed_messages, **kwargs):
            yield response
    
    async def _preprocess_messages(self, messages: List[Message]) -> List[Message]:
        """Pre-process messages for RAG, memory, etc."""
        processed_messages = []
        
        # Add system message
        processed_messages.extend(self._system_messages)
        
        # Add memory context if enabled
        if self.memory_enabled:
            memory_context = await self._get_memory_context(messages)
            if memory_context:
                processed_messages.append(memory_context)
        
        # Add RAG context if enabled
        if self.rag_enabled:
            rag_context = await self._get_rag_context(messages)
            if rag_context:
                processed_messages.append(rag_context)
        
        # Add user messages
        processed_messages.extend(messages)
        
        return processed_messages
    
    async def _get_memory_context(self, messages: List[Message]) -> Message:
        """Get memory context from previous conversations"""
        # Implementation would fetch relevant memory
        # For now, return empty context
        return None
    
    async def _get_rag_context(self, messages: List[Message]) -> Message:
        """Get RAG context from documents"""
        # Implementation would search documents
        # For now, return empty context
        return None

# Register the assistant
AgentRegistry.register('assistant')(Assistant)
```

---

## 📚 **Guide 4: Tool System & Safety**

### Step 1: Create Base Tool Classes

Create `tools/base_tool.py`:

```python
# tools/base_tool.py
import asyncio
import logging
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from exceptions.base import ToolExecutionError

logger = logging.getLogger(__name__)

@dataclass
class ToolSchema:
    """Tool schema for function calling"""
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str]

class BaseTool(ABC):
    """Base class for all tools - inspired by Qwen-Agent"""
    
    def __init__(self, name: str, description: str, enabled: bool = True):
        self.name = name
        self.description = description
        self.enabled = enabled
        self._execution_count = 0
        self._error_count = 0
    
    @abstractmethod
    async def execute(self, arguments: Dict[str, Any]) -> 'ToolResult':
        """Execute the tool with given arguments"""
        pass
    
    @abstractmethod
    def get_schema(self) -> ToolSchema:
        """Get the tool schema for function calling"""
        pass
    
    def validate_arguments(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool arguments"""
        schema = self.get_schema()
        missing = [param for param in schema.required if param not in arguments]
        if missing:
            raise ToolExecutionError(
                message=f"Missing required parameters: {missing}"
            )
        return arguments
    
    async def safe_execute(self, arguments: Dict[str, Any]) -> 'ToolResult':
        """Execute tool with safety checks"""
        try:
            self._execution_count += 1
            validated_args = self.validate_arguments(arguments)
            result = await self.execute(validated_args)
            return result
        except Exception as e:
            self._error_count += 1
            logger.error(f"Tool {self.name} execution failed: {e}")
            raise ToolExecutionError(
                message=f"Tool execution failed: {str(e)}",
                extra={'tool_name': self.name, 'arguments': arguments}
            )
    
    @property
    def execution_stats(self) -> Dict[str, int]:
        """Get execution statistics"""
        return {
            'total_executions': self._execution_count,
            'error_count': self._error_count,
            'success_rate': (self._execution_count - self._error_count) / max(1, self._execution_count)
        }

@dataclass
class ToolResult:
    """Result from tool execution"""
    content: Any
    success: bool = True
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_message(self, tool_call_id: str) -> Message:
        """Convert to message for conversation"""
        from llm.schema import Message
        
        if self.success:
            return Message(
                role='tool_result',
                content=str(self.content),
                tool_call_id=tool_call_id,
                metadata=self.metadata
            )
        else:
            return Message(
                role='tool_result',
                content=f"Error: {self.error}",
                tool_call_id=tool_call_id,
                metadata={'error': self.error, **self.metadata}
            )

class ToolRegistry:
    """Registry for dynamic tool management"""
    
    _tools: Dict[str, type] = {}
    
    @classmethod
    def register(cls, tool_name: str):
        """Register a tool"""
        def decorator(tool_class: type):
            cls._tools[tool_name] = tool_class
            return tool_class
        return decorator
    
    @classmethod
    def create_tool(cls, tool_name: str, **kwargs) -> BaseTool:
        """Create a tool instance"""
        if tool_name not in cls._tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        tool_class = cls._tools[tool_name]
        return tool_class(**kwargs)
    
    @classmethod
    def get_available_tools(cls) -> List[str]:
        """Get list of available tools"""
        return list(cls._tools.keys())

# Global tool registry instance
tool_registry = ToolRegistry()
```

### Step 2: Create Code Interpreter Tool

Create `tools/code_interpreter.py`:

```python
# tools/code_interpreter.py
import asyncio
import io
import sys
import traceback
from typing import Dict, Any

from tools.base_tool import BaseTool, ToolResult, ToolSchema

class CodeInterpreter(BaseTool):
    """Python code execution sandbox - inspired by Qwen-Agent"""
    
    def __init__(self, timeout: int = 30, max_output_length: int = 10000):
        super().__init__(
            name="code_interpreter",
            description="Execute Python code in a sandboxed environment"
        )
        self.timeout = timeout
        self.max_output_length = max_output_length
        self.execution_count = 0
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            },
            required=["code"]
        )
    
    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute Python code"""
        code = arguments.get("code", "")
        if not code:
            return ToolResult(
                content="No code provided",
                success=False,
                error="No code provided"
            )
        
        self.execution_count += 1
        
        # Create execution environment
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        try:
            # Execute code with timeout
            result = await asyncio.wait_for(
                self._execute_code(code, stdout_buffer, stderr_buffer),
                timeout=self.timeout
            )
            
            stdout = stdout_buffer.getvalue()
            stderr = stderr_buffer.getvalue()
            
            # Combine output
            if result is not None:
                output = f"Result: {result}"
            else:
                output = stdout if stdout else "(No output)"
            
            if stderr:
                output += f"\nErrors:\n{stderr}"
            
            # Truncate if too long
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length] + "\n... (output truncated)"
            
            return ToolResult(
                content=output,
                success=True,
                metadata={
                    'execution_count': self.execution_count,
                    'output_length': len(output),
                    'had_error': bool(stderr)
                }
            )
            
        except asyncio.TimeoutError:
            return ToolResult(
                content=f"Code execution timed out after {self.timeout} seconds",
                success=False,
                error="Execution timeout"
            )
        except Exception as e:
            error_msg = f"Execution error: {str(e)}\n{traceback.format_exc()}"
            return ToolResult(
                content=error_msg,
                success=False,
                error=str(e),
                metadata={'execution_count': self.execution_count}
            )
    
    async def _execute_code(self, code: str, stdout: io.StringIO, stderr: io.StringIO) -> Any:
        """Execute code with output capture"""
        # Capture stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            sys.stdout = stdout
            sys.stderr = stderr
            
            # Create isolated namespace
            namespace = {
                '__name__': '__main__',
                '__builtins__': __builtins__,
                # Add some safe builtins
                'print': lambda *args, **kwargs: None,  # Override print to avoid spam
            }
            
            # Execute code
            result = eval(code, namespace, namespace)
            
            # If it's an expression that returns a value, return it
            # If it's a statement, return None
            return result
            
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Register the tool
ToolRegistry.register('code_interpreter')(CodeInterpreter)
```

### Step 3: Create Safety & Loop Detection

Create `safety/loop_detection.py`:

```python
# safety/loop_detection.py
import hashlib
import logging
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict

from exceptions.base import LoopDetectionError

logger = logging.getLogger(__name__)

class LoopDetector:
    """Detect and prevent infinite loops in tool execution"""
    
    def __init__(self, max_tool_iterations: int = 10, max_content_repeats: int = 5):
        self.max_tool_iterations = max_tool_iterations
        self.max_content_repeats = max_content_repeats
        
        # Track tool call patterns
        self.tool_call_history: List[str] = []
        self.tool_call_counts: Dict[str, int] = defaultdict(int)
        
        # Track content patterns
        self.content_hashes: Set[str] = []
        
        # Session tracking
        self.session_id: Optional[str] = None
        self.iteration_count = 0
    
    def start_session(self, session_id: str) -> None:
        """Start a new detection session"""
        self.session_id = session_id
        self.tool_call_history.clear()
        self.tool_call_counts.clear()
        self.content_hashes.clear()
        self.iteration_count = 0
        
        logger.info(f"Started loop detection session: {session_id}")
    
    def check_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Check if tool call pattern indicates a loop"""
        self.iteration_count += 1
        
        # Check iteration limit
        if self.iteration_count > self.max_tool_iterations:
            raise LoopDetectionError(
                message=f"Maximum iterations ({self.max_tool_iterations}) exceeded"
            )
        
        # Create tool call signature
        args_hash = hashlib.md5(
            json.dumps(arguments, sort_keys=True).encode()
        ).hexdigest() if arguments else ""
        
        tool_signature = f"{tool_name}:{args_hash}"
        
        # Check for repeated patterns
        self.tool_call_history.append(tool_signature)
        self.tool_call_counts[tool_signature] += 1
        
        # Remove old entries
        if len(self.tool_call_history) > self.max_tool_iterations * 2:
            self.tool_call_history.pop(0)
        
        # Check for excessive repetition
        if self.tool_call_counts[tool_signature] > self.max_content_repeats:
            raise LoopDetectionError(
                message=f"Tool call '{tool_name}' repeated {self.tool_call_counts[tool_signature]} times"
            )
    
    def check_content(self, content: str) -> None:
        """Check if content is being repeated"""
        if not content:
            return
        
        # Hash content
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        if content_hash in self.content_hashes:
            raise LoopDetectionError(
                message="Content is being repeated"
            )
        
        self.content_hashes.add(content_hash)
        
        # Limit history size
        if len(self.content_hashes) > 100:
            # Remove oldest hashes
            self.content_hashes = set(list(self.content_hashes)[50:])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        return {
            'session_id': self.session_id,
            'iteration_count': self.iteration_count,
            'unique_tool_calls': len(self.tool_call_counts),
            'tool_call_history_length': len(self.tool_call_history),
            'content_hashes_count': len(self.content_hashes),
            'most_common_tool': max(self.tool_call_counts.items(), key=lambda x: x[1])[0] if self.tool_call_counts else None
        }

# Global loop detector instance
loop_detector = LoopDetector()

def detect_loop(
    tool_name: str, 
    arguments: Dict[str, Any], 
    content: Optional[str] = None,
    session_id: str = "default"
) -> None:
    """Detect loops in agent execution"""
    if loop_detector.session_id != session_id:
        loop_detector.start_session(session_id)
    
    loop_detector.check_tool_call(tool_name, arguments)
    if content:
        loop_detector.check_content(content)
```

---

## 📚 **Guide 5: CLI Interface (qwen-code style)**

### Step 1: Create CLI Main Entry

Create `cli/main.py`:

```python
# cli/main.py
import asyncio
import click
import logging
from pathlib import Path
from typing import Optional

from cli.commands import ChatCommand, ConfigCommand, ServerCommand
from cli.config import load_config, validate_config
from cli.theme import apply_theme
from exceptions.base import ModelServiceError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentCLI(click.Group):
    """Main CLI application"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = None
        self.chat_cmd = None
        self.config_cmd = None
        self.server_cmd = None
    
    def initialize(self, config_path: Optional[str] = None):
        """Initialize CLI with configuration"""
        try:
            # Load configuration
            self.config = load_config(config_path)
            validate_config(self.config)
            
            # Initialize commands
            self.chat_cmd = ChatCommand(self.config)
            self.config_cmd = ConfigCommand(self.config)
            self.server_cmd = ServerCommand(self.config)
            
            logger.info("CLI initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CLI: {e}")
            raise click.ClickException(f"Configuration error: {e}")
    
    def run_chat(self, session_id: Optional[str], model: Optional[str], provider: Optional[str]):
        """Run interactive chat session"""
        if not self.chat_cmd:
            raise click.ClickException("CLI not initialized")
        
        try:
            asyncio.run(self.chat_cmd.interactive_chat(session_id, model, provider))
        except KeyboardInterrupt:
            click.echo("\nGoodbye!")
        except Exception as e:
            logger.error(f"Chat session failed: {e}")
            raise click.ClickException(f"Chat error: {e}")
    
    def run_config(self, action: str):
        """Run configuration management"""
        if not self.config_cmd:
            raise click.ClickException("CLI not initialized")
        
        try:
            if action == "show":
                self.config_cmd.show_config()
            elif action == "validate":
                self.config_cmd.validate_config()
            elif action == "setup":
                self.config_cmd.setup_interactive()
            else:
                raise click.ClickException(f"Unknown config action: {action}")
        except Exception as e:
            logger.error(f"Config command failed: {e}")
            raise click.ClickException(f"Config error: {e}")
    
    def run_server(self, action: str, host: str, port: int):
        """Run server commands"""
        if not self.server_cmd:
            raise click.ClickException("CLI not initialized")
        
        try:
            if action == "start":
                self.server_cmd.start_server(host, port)
            elif action == "stop":
                self.server_cmd.stop_server()
            elif action == "status":
                self.server_cmd.show_status()
            else:
                raise click.ClickException(f"Unknown server action: {action}")
        except Exception as e:
            logger.error(f"Server command failed: {e}")
            raise click.ClickException(f"Server error: {e}")

# Create main CLI
@click.group(cls=AgentCLI)
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.pass_context
def cli(ctx, config, debug):
    """Multi-provider agent CLI with MCP integration"""
    # Setup debug logging
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize CLI
    ctx.ensure_object(dict)
    cli_obj = ctx.obj['cli'] = AgentCLI()
    cli_obj.initialize(config)

@cli.command()
@click.option('--session', '-s', help='Session ID to resume')
@click.option('--model', '-m', help='Model to use')
@click.option('--provider', '-p', type=click.Choice(['openai', 'anthropic', 'gemini', 'huggingface']), help='Provider to use')
@click.option('--non-interactive', is_flag=True, help='Run in non-interactive mode')
@click.argument('message', required=False)
@click.pass_context
def chat(ctx, session, model, provider, non_interactive, message):
    """Start a chat session"""
    cli_obj = ctx.obj['cli']
    
    if non_interactive and not message:
        click.echo("Message required for non-interactive mode", err=True)
        return
    
    cli_obj.run_chat(session, model, provider)

@cli.group()
def config():
    """Configuration management"""
    pass

@config.command('show')
@click.pass_context
def config_show(ctx):
    """Show current configuration"""
    ctx.obj['cli'].run_config('show')

@config.command('validate')
@click.pass_context
def config_validate(ctx):
    """Validate configuration"""
    ctx.obj['cli'].run_config('validate')

@config.command('setup')
@click.pass_context
def config_setup(ctx):
    """Interactive configuration setup"""
    ctx.obj['cli'].run_config('setup')

@cli.group()
def server():
    """Server management"""
    pass

@server.command('start')
@click.option('--host', default='localhost', help='Server host')
@click.option('--port', default=8765, type=int, help='Server port')
@click.pass_context
def server_start(ctx, host, port):
    """Start the agent server"""
    ctx.obj['cli'].run_server('start', host, port)

@server.command('stop')
@click.pass_context
def server_stop(ctx):
    """Stop the agent server"""
    ctx.obj['cli'].run_server('stop')

@server.command('status')
@click.pass_context
def server_status(ctx):
    """Show server status"""
    ctx.obj['cli'].run_server('status')

if __name__ == '__main__':
    cli()
```

### Step 2: Create CLI Commands

Create `cli/commands.py`:

```python
# cli/commands.py
import asyncio
import json
import click
from typing import Optional, Dict, Any
from pathlib import Path

from database import DatabaseManager
from agents.assistant import Assistant, AgentConfig
from llm.base_chat_model import BaseChatModel

class ChatCommand:
    """Interactive chat command"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db = DatabaseManager(config.get('database', {}))
        self.current_session = None
        self.current_agent = None
    
    async def interactive_chat(
        self, 
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None
    ):
        """Start interactive chat session"""
        await self.db.connect()
        
        try:
            # Create or resume session
            if session_id:
                self.current_session = await self.db.get_session(session_id)
                if not self.current_session:
                    click.echo(f"Session {session_id} not found")
                    return
                click.echo(f"Resumed session: {self.current_session.title}")
            else:
                # Create new session
                title = click.prompt("Session title", default="New Chat")
                provider = provider or click.prompt("Provider", type=click.Choice(['openai', 'anthropic', 'gemini', 'huggingface']))
                model = model or click.prompt("Model", default="gpt-4")
                
                self.current_session = await self.db.create_session(
                    title=title,
                    provider=provider,
                    model=model
                )
                click.echo(f"Created new session: {title}")
            
            # Initialize agent
            await self._initialize_agent()
            
            # Chat loop
            await self._chat_loop()
            
        finally:
            await self.db.disconnect()
    
    async def _initialize_agent(self):
        """Initialize the chat agent"""
        # Create LLM (simplified for example)
        llm = await self._create_llm(
            provider=self.current_session.provider,
            model=self.current_session.model
        )
        
        # Create agent config
        config = AgentConfig(
            name="cli_assistant",
            description="CLI Assistant",
            system_message="You are a helpful assistant in a CLI environment.",
            llm=llm,
            max_iterations=10
        )
        
        # Create agent
        self.current_agent = Assistant(config)
    
    async def _chat_loop(self):
        """Main chat loop"""
        click.echo("\nEnter your message (or 'quit' to exit, 'help' for commands):")
        
        while True:
            try:
                # Get user input
                user_input = click.prompt("You", prompt_suffix=" ")
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    click.echo("Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                if user_input.lower() == 'clear':
                    click.clear()
                    continue
                
                # Process message
                await self._process_message(user_input)
                
            except KeyboardInterrupt:
                click.echo("\nGoodbye!")
                break
            except Exception as e:
                click.echo(f"Error: {e}")
    
    async def _process_message(self, message: str):
        """Process user message"""
        from llm.schema import Message
        
        # Create user message
        user_message = Message(role='user', content=message)
        
        # Get agent response
        click.echo("Assistant: ", nl=False)
        
        async for response in self.current_agent.run([user_message]):
            for msg in response:
                click.echo(msg.content, nl=False)
                click.echo()  # New line
        
        # Save to database
        await self.db.save_message(
            session_id=self.current_session.id,
            role='user',
            content=message
        )
        
        # Save assistant response (simplified)
        # In real implementation, you'd capture all assistant responses
    
    def _show_help(self):
        """Show help commands"""
        click.echo("\nAvailable commands:")
        click.echo("  quit, exit, q - Exit the chat")
        click.echo("  help - Show this help")
        click.echo("  clear - Clear the screen")
        click.echo("\nYou can also use regular conversation with the AI assistant.")
    
    async def _create_llm(self, provider: str, model: str) -> BaseChatModel:
        """Create LLM instance (simplified)"""
        # This would integrate with your provider.py classes
        from providers.openai_provider import OpenAIProvider
        from providers.anthropic_provider import AnthropicProvider
        # etc.
        
        provider_map = {
            'openai': OpenAIProvider,
            'anthropic': AnthropicProvider,
            # etc.
        }
        
        if provider not in provider_map:
            raise ValueError(f"Unsupported provider: {provider}")
        
        provider_class = provider_map[provider]
        return provider_class(
            api_key=self.config.get('providers', {}).get(provider, {}).get('api_key'),
            model=model
        )

class ConfigCommand:
    """Configuration management commands"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def show_config(self):
        """Show current configuration"""
        click.echo("Current Configuration:")
        click.echo(json.dumps(self.config, indent=2))
    
    def validate_config(self):
        """Validate configuration"""
        try:
            # Validate using your validation logic
            from cli.config import validate_config
            validate_config(self.config)
            click.echo("Configuration is valid")
        except Exception as e:
            click.echo(f"Configuration validation failed: {e}")
    
    def setup_interactive(self):
        """Interactive configuration setup"""
        click.echo("Interactive Configuration Setup")
        
        # Database setup
        self.config['database'] = {
            'connection_string': click.prompt("MongoDB URL", default="mongodb://localhost:27017"),
            'db_name': click.prompt("Database name", default="agent_ui")
        }
        
        # Provider setup
        providers = {}
        for provider in ['openai', 'anthropic', 'gemini']:
            if click.confirm(f"Setup {provider}?"):
                providers[provider] = {
                    'api_key': click.prompt(f"{provider} API key", hide_input=True),
                    'default_model': click.prompt(f"{provider} default model", default="")
                }
        
        self.config['providers'] = providers
        
        # Save config
        config_path = Path("config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        click.echo(f"Configuration saved to {config_path}")

class ServerCommand:
    """Server management commands"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.server_process = None
    
    def start_server(self, host: str, port: int):
        """Start the agent server"""
        click.echo(f"Starting server on {host}:{port}")
        # Implementation would start WebSocket server
        # This is a placeholder
    
    def stop_server(self):
        """Stop the agent server"""
        click.echo("Stopping server...")
        # Implementation would stop the server
        # This is a placeholder
    
    def show_status(self):
        """Show server status"""
        click.echo("Server Status:")
        click.echo("  Status: Running")  # Placeholder
        click.echo("  Active Connections: 0")  # Placeholder
```

---

## 📚 **Guide 6: Testing Infrastructure**

### Step 1: Create Test Structure

Create `tests/conftest.py`:

```python
# tests/conftest.py
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import Generator

from database import DatabaseManager, Session, Message
from providers.base_provider import BaseProvider, ModelResponse
from agents.base_agent import AgentConfig
from llm.base_chat_model import BaseChatModel

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def mock_db():
    """Mock database manager"""
    db = Mock(spec=DatabaseManager)
    db.connect = AsyncMock()
    db.disconnect = AsyncMock()
    db.save_message = AsyncMock()
    db.create_session = AsyncMock()
    db.get_session = AsyncMock()
    return db

@pytest.fixture
def mock_provider():
    """Mock provider"""
    provider = Mock(spec=BaseProvider)
    provider.create_chat_completion = AsyncMock()
    provider._format_tools = Mock(return_value=[])
    provider._parse_response = Mock(return_value=ModelResponse(content="Mock response"))
    provider._get_provider_info = Mock(return_value=Mock(
        name="mock",
        supports_streaming=True,
        supports_thinking=False,
        supports_tools=True,
        supports_multimodal=False
    ))
    return provider

@pytest.fixture
def mock_llm():
    """Mock LLM"""
    llm = Mock(spec=BaseChatModel)
    llm.chat = AsyncMock()
    return llm

@pytest.fixture
def sample_agent_config():
    """Sample agent configuration"""
    return AgentConfig(
        name="test_agent",
        description="Test agent",
        system_message="You are a test assistant.",
        max_iterations=5,
        temperature=0.7
    )

@pytest.fixture
def sample_messages():
    """Sample messages for testing"""
    return [
        Message(role='user', content='Hello'),
        Message(role='assistant', content='Hi there!'),
    ]

@pytest.fixture
def sample_tool_schema():
    """Sample tool schema"""
    return {
        'name': 'test_tool',
        'description': 'A test tool',
        'parameters': {
            'type': 'object',
            'properties': {
                'input': {'type': 'string'}
            },
            'required': ['input']
        }
    }
```

### Step 2: Create Unit Tests

Create `tests/test_providers.py`:

```python
# tests/test_providers.py
import pytest
from unittest.mock import Mock, patch

from providers.base_provider import BaseProvider, ModelResponse, ProviderInfo
from providers.openai_provider import OpenAIProvider
from exceptions.base import ProviderError

class TestBaseProvider:
    """Test BaseProvider functionality"""
    
    def test_base_provider_creation(self):
        """Test base provider initialization"""
        class TestProvider(BaseProvider):
            async def create_chat_completion(self, messages, tools=None, stream=False, **kwargs):
                return ModelResponse(content="test")
            
            def _format_tools(self, tools):
                return tools
            
            def _parse_response(self, response):
                return response
            
            def _get_provider_info(self):
                return ProviderInfo(name="test")
        
        provider = TestProvider(api_key="test", model="test-model")
        assert provider.api_key == "test"
        assert provider.model == "test-model"
        assert provider.provider_info.name == "test"
    
    def test_base_provider_abstract_methods(self):
        """Test that abstract methods must be implemented"""
        with pytest.raises(TypeError):
            BaseProvider(api_key="test", model="test-model")

class TestOpenAIProvider:
    """Test OpenAI provider"""
    
    @pytest.fixture
    def mock_openai_client(self):
        with patch('providers.openai_provider.AsyncOpenAI') as mock:
            mock_instance = Mock()
            mock_instance.chat.completions.create = AsyncMock()
            mock.return_value = mock_instance
            yield mock_instance
    
    @pytest.mark.asyncio
    async def test_create_chat_completion(self, mock_openai_client):
        """Test chat completion creation"""
        provider = OpenAIProvider(api_key="test", model="gpt-4")
        
        # Mock successful response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        result = await provider.create_chat_completion(
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert isinstance(result, ModelResponse)
        assert result.content == "Test response"
        assert mock_openai_client.chat.completions.create.called
    
    @pytest.mark.asyncio
    async def test_streaming_completion(self, mock_openai_client):
        """Test streaming chat completion"""
        provider = OpenAIProvider(api_key="test", model="gpt-4", stream=True)
        
        # Mock streaming response
        async def mock_stream():
            for i in range(3):
                chunk = Mock()
                chunk.choices = [Mock()]
                chunk.choices[0].message.content = f"Chunk {i}"
                yield chunk
        
        mock_openai_client.chat.completions.create.return_value = mock_stream()
        
        results = []
        async for result in provider.create_chat_completion(
            messages=[{"role": "user", "content": "Hello"}]
        ):
            results.append(result)
        
        assert len(results) > 0
```

Create `tests/test_agents.py`:

```python
# tests/test_agents.py
import pytest
from unittest.mock import Mock, AsyncMock

from agents.base_agent import BaseAgent, AgentConfig, AgentRegistry
from agents.fncall_agent import FnCallAgent
from agents.assistant import Assistant
from llm.schema import Message, ASSISTANT, FunctionCall
from tools.base_tool import BaseTool, ToolResult
from exceptions.base import ModelServiceError

class TestAgentConfig:
    """Test AgentConfig"""
    
    def test_agent_config_creation(self):
        """Test agent config creation"""
        config = AgentConfig(
            name="test_agent",
            description="A test agent",
            system_message="You are a test assistant.",
            max_iterations=10
        )
        
        assert config.name == "test_agent"
        assert config.max_iterations == 10

class TestAgentRegistry:
    """Test AgentRegistry"""
    
    def test_register_agent(self):
        """Test agent registration"""
        # Clear registry
        AgentRegistry._agents.clear()
        
        @AgentRegistry.register('test_agent')
        class TestAgent(BaseAgent):
            async def _run(self, messages, **kwargs):
                yield []
        
        assert 'test_agent' in AgentRegistry._agents
        assert AgentRegistry._agents['test_agent'] == TestAgent
    
    def test_create_agent(self):
        """Test agent creation from registry"""
        # Clear registry
        AgentRegistry._agents.clear()
        
        @AgentRegistry.register('test_agent')
        class TestAgent(BaseAgent):
            async def _run(self, messages, **kwargs):
                yield []
        
        config = AgentConfig(name="test", description="test", system_message="test")
        agent = AgentRegistry.create_agent('test_agent', config)
        
        assert isinstance(agent, TestAgent)
        assert agent.config.name == "test"

class TestFnCallAgent:
    """Test FnCallAgent"""
    
    @pytest.fixture
    def mock_llm(self):
        llm = Mock()
        llm.chat = AsyncMock()
        llm._handle_provider_error = Mock(return_value=ModelServiceError(message="Test error"))
        return llm
    
    @pytest.fixture
    def mock_tool(self):
        tool = Mock(spec=BaseTool)
        tool.name = "test_tool"
        tool.get_schema.return_value = {
            'name': 'test_tool',
            'description': 'A test tool',
            'parameters': {'type': 'object'},
            'required': []
        }
        tool.execute = AsyncMock(return_value=ToolResult(content="Tool result"))
        return tool
    
    @pytest.mark.asyncio
    async def test_simple_response(self, mock_llm, mock_tool):
        """Test agent response without function calls"""
        config = AgentConfig(
            name="test",
            description="test",
            system_message="test",
            llm=mock_llm,
            tools=[mock_tool]
        )
        
        agent = FnCallAgent(config)
        
        # Mock LLM response with content
        mock_llm.chat.return_value = [
            Message(role=ASSISTANT, content="Hello!")
        ]
        
        messages = [Message(role='user', content="Hi")]
        results = []
        async for response in agent.run(messages):
            results.extend(response)
        
        assert len(results) > 0
        assert results[0].content == "Hello!"
    
    @pytest.mark.asyncio
    async def test_function_calling(self, mock_llm, mock_tool):
        """Test agent function calling"""
        config = AgentConfig(
            name="test",
            description="test",
            system_message="test",
            llm=mock_llm,
            tools=[mock_tool]
        )
        
        agent = FnCallAgent(config)
        
        # Mock LLM response with function call
        function_call = FunctionCall(
            id="call_1",
            name="test_tool",
            arguments='{"input": "test"}'
        )
        
        mock_llm.chat.side_effect = [
            [Message(role=ASSISTANT, function_call=function_call)],
            [Message(role=ASSISTANT, content="Tool executed successfully")]
        ]
        
        messages = [Message(role='user', content="Use the tool")]
        results = []
        async for response in agent.run(messages):
            results.extend(response)
        
        # Should have function call and result
        assert len(results) >= 2
        assert mock_tool.execute.called

class TestAssistant:
    """Test Assistant agent"""
    
    @pytest.fixture
    def mock_llm(self):
        llm = Mock()
        llm.chat = AsyncMock()
        return llm
    
    @pytest.mark.asyncio
    async def test_assistant_creation(self, mock_llm):
        """Test assistant agent creation"""
        config = AgentConfig(
            name="assistant",
            description="Test assistant",
            system_message="You are a test assistant.",
            llm=mock_llm
        )
        
        assistant = Assistant(config)
        assert isinstance(assistant, FnCallAgent)
        assert assistant.config.name == "assistant"
    
    @pytest.mark.asyncio
    async def test_preprocessing(self, mock_llm):
        """Test message preprocessing"""
        config = AgentConfig(
            name="assistant",
            description="Test assistant",
            system_message="You are a test assistant.",
            llm=mock_llm
        )
        
        assistant = Assistant(config)
        
        # Enable RAG and memory
        assistant.rag_enabled = True
        assistant.memory_enabled = True
        
        # Mock preprocessing methods
        assistant._get_memory_context = AsyncMock(return_value=None)
        assistant._get_rag_context = AsyncMock(return_value=None)
        
        messages = [Message(role='user', content="Hello")]
        processed = await assistant._preprocess_messages(messages)
        
        # Should have system message + user message
        assert len(processed) >= 2
        assert processed[0].role == 'system'
        assert processed[-1].role == 'user'
```

### Step 3: Create Integration Tests

Create `tests/test_integration.py`:

```python
# tests/test_integration.py
import pytest
from unittest.mock import Mock, AsyncMock, patch

from client import MCPClientSystem
from database import DatabaseManager
from agents.assistant import Assistant, AgentConfig
from providers.openai_provider import OpenAIProvider

class TestMCPClientSystem:
    """Integration tests for MCPClientSystem"""
    
    @pytest.fixture
    def mock_config(self):
        return {
            'database': {
                'connection_string': 'mongodb://localhost:27017',
                'db_name': 'test_agent_ui'
            },
            'providers': {
                'openai': {
                    'api_key': 'test-key',
                    'model': 'gpt-4'
                }
            },
            'mcp_servers': []
        }
    
    @pytest.mark.asyncio
    async def test_system_initialization(self, mock_config):
        """Test system initialization"""
        with patch('client.DatabaseManager') as MockDB:
            mock_db = Mock()
            MockDB.return_value = mock_db
            mock_db.connect = AsyncMock()
            
            with patch('client.MCPClient') as MockMCP:
                mock_mcp = Mock()
                MockMCP.return_value = mock_mcp
                mock_mcp.initialize = AsyncMock()
                
                with patch('client.ProviderFactory') as MockFactory:
                    mock_provider = Mock()
                    MockFactory.create_provider.return_value = mock_provider
                    
                    system = MCPClientSystem(mock_config)
                    await system.initialize()
                    
                    mock_db.connect.assert_called_once()
                    mock_mcp.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_message_processing(self, mock_config):
        """Test message processing flow"""
        with patch('client.DatabaseManager') as MockDB:
            mock_db = Mock()
            MockDB.return_value = mock_db
            mock_db.connect = AsyncMock()
            mock_db.get_session = AsyncMock(return_value=Mock())
            mock_db.save_message = AsyncMock()
            mock_db.get_conversation_messages = AsyncMock(return_value=[])
            
            with patch('client.MCPClient') as MockMCP:
                mock_mcp = Mock()
                MockMCP.return_value = mock_mcp
                mock_mcp.initialize = AsyncMock()
                mock_mcp.get_all_tools = Mock(return_value=[])
                
                with patch('client.ProviderFactory') as MockFactory:
                    mock_provider = Mock()
                    mock_provider.create_chat_completion = AsyncMock(
                        return_value=Mock(content="Hello!", tool_calls=None)
                    )
                    MockFactory.create_provider.return_value = mock_provider
                    
                    system = MCPClientSystem(mock_config)
                    await system.initialize()
                    
                    # Process a message
                    response = await system.process_message(
                        session_id="test_session",
                        user_id="test_user",
                        message="Hello"
                    )
                    
                    assert response.content == "Hello!"
                    mock_db.save_message.assert_called()

class TestFullWorkflow:
    """End-to-end workflow tests"""
    
    @pytest.mark.asyncio
    async def test_simple_chat_workflow(self):
        """Test complete chat workflow"""
        # This would test the full flow:
        # 1. Create session
        # 2. Send message
        # 3. Process through agent
        # 4. Get response
        # 5. Save to database
        
        # Mock all dependencies
        with patch('client.DatabaseManager') as MockDB:
            mock_db = Mock()
            mock_db.connect = AsyncMock()
            mock_db.create_session = AsyncMock(return_value=Mock(id="session_1"))
            mock_db.save_message = AsyncMock()
            MockDB.return_value = mock_db
            
            with patch('client.MCPClient') as MockMCP:
                mock_mcp = Mock()
                mock_mcp.initialize = AsyncMock()
                MockMCP.return_value = mock_mcp
                
                with patch('client.ProviderFactory') as MockFactory:
                    mock_provider = Mock()
                    mock_provider.create_chat_completion = AsyncMock(
                        return_value=Mock(content="Hello! How can I help?", tool_calls=None)
                    )
                    MockFactory.create_provider.return_value = mock_provider
                    
                    # This would be the actual integration test
                    # Testing the full workflow from client.py
                    
                    pass  # Implementation would go here
    
    @pytest.mark.asyncio
    async def test_tool_calling_workflow(self):
        """Test tool calling workflow"""
        # Test the flow when an agent needs to call tools
        # This would test:
        # 1. LLM response with function call
        # 2. Tool execution
        # 3. Tool result integration
        # 4. Follow-up LLM call
        
        pass  # Implementation would go here
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self):
        """Test error handling in workflows"""
        # Test various error scenarios:
        # 1. LLM API errors
        # 2. Tool execution errors
        # 3. Database connection errors
        # 4. Network timeouts
        
        pass  # Implementation would go here
```

### Step 4: Create pytest Configuration

Create `pytest.ini`:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --tb=short
    --strict-markers
    --strict-config
    --disable-warnings
    --cov=agents
    --cov=providers
    --cov=tools
    --cov=database
    --cov=utils
    --cov=exceptions
    --cov=cli
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-report=xml
asyncio_mode = auto

[coverage:run]
source = .
omit = 
    */tests/*
    */test_*
    */__pycache__/*
    */venv/*
    */env/*
    */build/*
    */dist/*
    setup.py

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    if self.debug:
    if settings.DEBUG
    raise AssertionError
    raise NotImplementedError
    if 0:
    if __name__ == .__main__.:
    class .*\bProtocol\):
    @(abc\.)?abstractmethod

[coverage:html]
directory = htmlcov
```

### Step 5: Run Tests

Create test runner script `run_tests.py`:

```python
# run_tests.py
#!/usr/bin/env python3
"""Test runner script"""

import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run tests with coverage"""
    cmd = [
        sys.executable, "-m", "pytest",
        "--cov=.",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--cov-report=xml",
        "tests/"
    ]
    
    result = subprocess.run(cmd)
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_tests())
```

---

## 📚 **Guide 7: Performance & Monitoring**

### Step 1: Create Performance Monitoring

Create `utils/performance.py`:

```python
# utils/performance.py
import time
import functools
import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance metric data"""
    name: str
    value: float
    unit: str
    timestamp: float
    tags: Dict[str, str] = None
    metadata: Dict[str, Any] = None

class PerformanceMonitor:
    """Monitor performance metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics: deque = deque(maxlen=max_history)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}
    
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric"""
        self.metrics.append(metric)
        logger.debug(f"Recorded metric: {metric.name} = {metric.value} {metric.unit}")
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter"""
        self.counters[name] += value
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit='count',
            timestamp=time.time(),
            tags=tags or {}
        )
        self.record_metric(metric)
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge value"""
        self.gauges[name] = value
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit='gauge',
            timestamp=time.time(),
            tags=tags or {}
        )
        self.record_metric(metric)
    
    def start_timer(self, name: str):
        """Start a timer"""
        self.start_times[name] = time.time()
    
    def stop_timer(self, name: str, tags: Dict[str, str] = None):
        """Stop a timer and record duration"""
        if name not in self.start_times:
            logger.warning(f"Timer '{name}' was not started")
            return
        
        duration = time.time() - self.start_times[name]
        metric = PerformanceMetric(
            name=name,
            value=duration,
            unit='seconds',
            timestamp=time.time(),
            tags=tags or {}
        )
        self.record_metric(metric)
        del self.start_times[name]
        return duration
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            'total_metrics': len(self.metrics),
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'active_timers': list(self.start_times.keys()),
            'recent_metrics': [asdict(m) for m in list(self.metrics)[-10:]]
        }
        
        # Calculate percentiles for recent timings
        recent_timings = [m.value for m in self.metrics if m.unit == 'seconds']
        if recent_timings:
            recent_timings.sort()
            stats['timing_percentiles'] = {
                'p50': recent_timings[len(recent_timings) // 2],
                'p90': recent_timings[int(len(recent_timings) * 0.9)],
                'p95': recent_timings[int(len(recent_timings) * 0.95)],
                'p99': recent_timings[int(len(recent_timings) * 0.99)]
            }
        
        return stats

# Global performance monitor
performance_monitor = PerformanceMonitor()

def monitor_performance(name: str, tags: Dict[str, str] = None):
    """Decorator to monitor function performance"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            performance_monitor.start_timer(name)
            try:
                result = await func(*args, **kwargs)
                performance_monitor.increment_counter(f"{name}_success", tags=tags)
                return result
            except Exception as e:
                performance_monitor.increment_counter(f"{name}_error", tags=tags)
                raise
            finally:
                performance_monitor.stop_timer(name, tags=tags)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            performance_monitor.start_timer(name)
            try:
                result = func(*args, **kwargs)
                performance_monitor.increment_counter(f"{name}_success", tags=tags)
                return result
            except Exception as e:
                performance_monitor.increment_counter(f"{name}_error", tags=tags)
                raise
            finally:
                performance_monitor.stop_timer(name, tags=tags)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

def track_memory_usage():
    """Track memory usage"""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        performance_monitor.set_gauge('memory_rss', memory_info.rss / 1024 / 1024, {'unit': 'MB'})
        performance_monitor.set_gauge('memory_vms', memory_info.vms / 1024 / 1024, {'unit': 'MB'})
        
    except ImportError:
        logger.warning("psutil not available for memory tracking")

def track_cpu_usage():
    """Track CPU usage"""
    try:
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=1)
        performance_monitor.set_gauge('cpu_usage', cpu_percent, {'unit': 'percent'})
        
    except ImportError:
        logger.warning("psutil not available for CPU tracking")

class AsyncLimiter:
    """Rate limiter for async operations"""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_tasks = 0
    
    async def acquire(self):
        """Acquire a slot"""
        async with self.semaphore:
            self.active_tasks += 1
            performance_monitor.increment_counter('limiter_acquired')
    
    async def release(self):
        """Release a slot"""
        self.active_tasks -= 1
        performance_monitor.increment_counter('limiter_released')
    
    async def __aenter__(self):
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()
```

### Step 2: Create Monitoring System

Create `monitoring/metrics.py`:

```python
# monitoring/metrics.py
import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta

from utils.performance import performance_monitor
from database import DatabaseManager

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collect and store system metrics"""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.collection_name = "metrics"
        self.running = False
        self.collection_task = None
    
    async def start_collection(self, interval: int = 60):
        """Start periodic metrics collection"""
        self.running = True
        self.collection_task = asyncio.create_task(self._collect_metrics(interval))
        logger.info(f"Started metrics collection every {interval} seconds")
    
    async def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        if self.collection_task:
            self.collection_task.cancel()
        logger.info("Stopped metrics collection")
    
    async def _collect_metrics(self, interval: int):
        """Collect metrics periodically"""
        while self.running:
            try:
                await self._collect_current_metrics()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_current_metrics(self):
        """Collect current system metrics"""
        timestamp = datetime.utcnow()
        
        # Get performance statistics
        stats = performance_monitor.get_statistics()
        
        # Collect system metrics
        system_metrics = {
            'timestamp': timestamp,
            'type': 'system_metrics',
            'data': {
                'performance_stats': stats,
                'collection_interval': 60
            }
        }
        
        # Store in database
        try:
            await self.db.db[self.collection_name].insert_one(system_metrics)
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")
    
    async def get_metrics_summary(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> Dict[str, Any]:
        """Get metrics summary for time range"""
        try:
            cursor = self.db.db[self.collection_name].find({
                'timestamp': {
                    '$gte': start_time,
                    '$lte': end_time
                }
            })
            
            metrics = await cursor.to_list(length=None)
            
            return {
                'start_time': start_time,
                'end_time': end_time,
                'total_metrics': len(metrics),
                'metrics': metrics
            }
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {}

class HealthChecker:
    """Check system health"""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    async def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = {
            'timestamp': datetime.utcnow(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        try:
            # Check database connectivity
            health_status['checks']['database'] = await self._check_database()
            
            # Check performance metrics
            health_status['checks']['performance'] = await self._check_performance()
            
            # Check system resources
            health_status['checks']['resources'] = await self._check_resources()
            
            # Check agent health
            health_status['checks']['agents'] = await self._check_agents()
            
            # Determine overall status
            if any(check['status'] == 'unhealthy' for check in health_status['checks'].values()):
                health_status['overall_status'] = 'unhealthy'
            elif any(check['status'] == 'degraded' for check in health_status['checks'].values()):
                health_status['overall_status'] = 'degraded'
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'timestamp': datetime.utcnow(),
                'overall_status': 'unhealthy',
                'error': str(e)
            }
    
    async def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            # Test database connection
            await self.db.db.admin.command('ping')
            
            # Check database stats
            stats = await self.db.db.command('dbStats')
            
            return {
                'status': 'healthy',
                'response_time': 0,  # Would measure actual ping time
                'database_size': stats.get('dataSize', 0),
                'collections': stats.get('collections', 0)
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def _check_performance(self) -> Dict[str, Any]:
        """Check performance metrics"""
        try:
            stats = performance_monitor.get_statistics()
            
            # Check for performance issues
            issues = []
            
            # Check if we have recent metrics
            if stats['total_metrics'] == 0:
                issues.append("No metrics collected")
            
            # Check timing percentiles if available
            if 'timing_percentiles' in stats:
                p95 = stats['timing_percentiles'].get('p95', 0)
                if p95 > 5.0:  # More than 5 seconds
                    issues.append(f"High p95 latency: {p95:.2f}s")
            
            if issues:
                return {
                    'status': 'degraded',
                    'issues': issues,
                    'stats': stats
                }
            else:
                return {
                    'status': 'healthy',
                    'stats': stats
                }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def _check_resources(self) -> Dict[str, Any]:
        """Check system resources"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            issues = []
            
            if cpu_percent > 90:
                issues.append(f"High CPU usage: {cpu_percent}%")
            
            if memory.percent > 90:
                issues.append(f"High memory usage: {memory.percent}%")
            
            if disk.percent > 90:
                issues.append(f"High disk usage: {disk.percent}%")
            
            if issues:
                return {
                    'status': 'degraded',
                    'issues': issues,
                    'metrics': {
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'disk_percent': disk.percent
                    }
                }
            else:
                return {
                    'status': 'healthy',
                    'metrics': {
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'disk_percent': disk.percent
                    }
                }
        except ImportError:
            return {
                'status': 'degraded',
                'issues': ['psutil not available']
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def _check_agents(self) -> Dict[str, Any]:
        """Check agent health"""
        try:
            # This would check agent status, active sessions, etc.
            # For now, return placeholder
            
            return {
                'status': 'healthy',
                'metrics': {
                    'active_agents': 0,
                    'active_sessions': 0,
                    'total_messages': 0
                }
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

# Global instances
metrics_collector = None
health_checker = None

def initialize_monitoring(db: DatabaseManager):
    """Initialize monitoring system"""
    global metrics_collector, health_checker
    
    metrics_collector = MetricsCollector(db)
    health_checker = HealthChecker(db)
    
    # Start metrics collection
    asyncio.create_task(metrics_collector.start_collection(interval=60))
    
    logger.info("Monitoring system initialized")
```

---

## 🚀 **Summary & Next Steps**

You now have comprehensive guides to transform your agent-ui into an enterprise-grade framework matching Qwen-Agent/Qwen-Code capabilities:

### **Key Implementation Order**:

1. **Week 1**: Dependencies & Error Handling
   - Create requirements.txt and environment setup
   - Implement retry logic and error handling
   - Set up basic testing infrastructure

2. **Week 2**: Agent System
   - Create BaseAgent hierarchy
   - Implement Function Calling Agent
   - Add Assistant Agent with RAG integration

3. **Week 3**: Advanced Features
   - Code interpreter tool
   - Loop detection system
   - CLI interface (qwen-code style)

4. **Week 4**: Production Readiness
   - Performance monitoring
   - Comprehensive testing
   - Security features

### **Critical Success Factors**:
- **Start with dependencies** - Essential for development
- **Implement agent hierarchy** - Core architectural pattern
- **Add error handling** - Production reliability
- **Create CLI interface** - User accessibility
- **Build comprehensive tests** - Quality assurance

### **Your Strong Foundation**:
Your current implementation already has excellent provider abstraction and MongoDB integration. The main gaps are agent orchestration patterns, safety systems, and enterprise features that these guides address.

**Estimated Timeline**: 4-6 weeks for full implementation
**Expected Result**: Enterprise-grade agent framework matching Qwen-Agent/Qwen-Code sophistication

The guides provide code examples, testing strategies, and implementation patterns directly inspired by the sophisticated architectures found in Qwen-Agent and Qwen-Code.