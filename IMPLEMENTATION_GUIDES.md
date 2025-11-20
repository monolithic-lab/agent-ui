# Implementation Guides for Agent-UI

## 🎯 Quick Start Guide: Transform to Qwen-Level Framework

---

## 📋 Guide 1: Implement Tool Registry Pattern

**Priority**: CRITICAL  
**Time**: 2-3 hours  
**Impact**: Foundation for all tool-related features

### Why This Pattern Matters
Qwen-Agent uses a registry pattern that enables:
- Dynamic tool discovery and registration
- Tool hot-swapping without restart
- Centralized tool management
- Plugin system support

### Implementation Steps

#### Step 1: Update `tools/__init__.py`
```python
# tools/__init__.py
from .base_tool import BaseTool, ToolResult, ToolSchema
from .code_interpreter import CodeInterpreter

# Registry pattern - add this
TOOL_REGISTRY = {}

def register_tool(name: str):
    def decorator(cls):
        TOOL_REGISTRY[name] = cls
        return cls
    return decorator

def get_tool_registry():
    return TOOL_REGISTRY.copy()

def create_tool(name: str, **kwargs):
    """Factory function to create tools by name"""
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Tool '{name}' not found in registry")
    
    tool_class = TOOL_REGISTRY[name]
    return tool_class(**kwargs)
```

#### Step 2: Update BaseTool with Registry Support
```python
# tools/base_tool.py - Add registry support
from tools import register_tool, TOOL_REGISTRY

class BaseTool(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Auto-register subclasses
        if hasattr(cls, '__tool_name__'):
            TOOL_REGISTRY[cls.__tool_name__] = cls

@register_tool('tool_name')
class MyCustomTool(BaseTool):
    __tool_name__ = 'my_tool'  # Explicit registration
```

#### Step 3: Convert Existing Tools
```python
# tools/code_interpreter.py - Convert to registry pattern
from tools import register_tool

@register_tool('code_interpreter')  # Add this decorator
class CodeInterpreter(BaseTool):
    __tool_name__ = 'code_interpreter'
    
    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(
            name='code_interpreter',  # Use registry name
            description='Python code sandbox for executing Python code',
            enabled=True
        )
        # ... rest of implementation
```

### Testing the Registry
```python
# test_tool_registry.py
from tools import TOOL_REGISTRY, create_tool

def test_registry():
    print(f"Available tools: {list(TOOL_REGISTRY.keys())}")
    
    # Create tool from registry
    tool = create_tool('code_interpreter')
    assert tool.name == 'code_interpreter'
    
    print("✅ Tool registry working!")
```

---

## 📋 Guide 2: Enhanced Agent Registry System

**Priority**: CRITICAL  
**Time**: 3-4 hours  
**Impact**: Enables dynamic agent management

### Implementation Pattern (from Qwen-Agent)

#### Step 1: Create Agent Registry
```python
# agents/__init__.py
from .base_agent import BaseAgent, AgentConfig
from .fncall_agent import FnCallAgent
from .assistant import Assistant

# Agent Registry
AGENT_REGISTRY = {}

def register_agent(name: str):
    def decorator(cls):
        AGENT_REGISTRY[name] = cls
        return cls
    return decorator

def get_agent_registry():
    return AGENT_REGISTRY.copy()

def create_agent(name: str, config: AgentConfig):
    """Factory function to create agents by name"""
    if name not in AGENT_REGISTRY:
        raise ValueError(f"Agent '{name}' not found in registry")
    
    agent_class = AGENT_REGISTRY[name]
    return agent_class(config)
```

#### Step 2: Enhance BaseAgent with Registry
```python
# agents/base_agent.py - Add registry support
from agents import register_agent, AGENT_REGISTRY

class BaseAgent(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Auto-register subclasses
        if hasattr(cls, '__agent_name__'):
            AGENT_REGISTRY[cls.__agent_name__] = cls

@register_agent('base')
class BaseAgent(ABC):
    __agent_name__ = 'base'  # or use decorator parameter
    
    # ... existing implementation
```

#### Step 3: Convert Existing Agents
```python
# agents/assistant.py
from agents import register_agent

@register_agent('assistant')
class Assistant(FnCallAgent):
    __agent_name__ = 'assistant'
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        # ... existing implementation
```

### Agent Manager System
```python
# agents/manager.py
class AgentManager:
    def __init__(self):
        self.agents = {}  # name -> instance
        self.configs = {}  # name -> config
    
    def register_agent(self, name: str, config: AgentConfig):
        """Register and create agent"""
        self.configs[name] = config
        self.agents[name] = create_agent(name, config)
    
    def get_agent(self, name: str) -> BaseAgent:
        """Get agent instance"""
        if name not in self.agents:
            raise ValueError(f"Agent '{name}' not registered")
        return self.agents[name]
    
    def list_agents(self):
        """List all registered agents"""
        return list(self.agents.keys())
```

---

## 📋 Guide 3: Multi-Agent Coordination System

**Priority**: HIGH  
**Time**: 6-8 hours  
**Impact**: Enables group chat and agent collaboration

### Implementation Pattern (from Qwen-Agent group_chat.py)

#### Step 1: Group Chat Agent
```python
# agents/group_chat.py
from typing import List, Dict, Optional
from agents import BaseAgent, AgentConfig
from llm.schema import Message

class GroupChat(BaseAgent):
    """Multi-agent coordination system"""
    
    def __init__(
        self, 
        config: AgentConfig,
        agents: List[BaseAgent],
        selection_method: str = 'auto'
    ):
        super().__init__(config)
        self.agents = agents
        self.selection_method = selection_method
        self._agent_names = [agent.config.name for agent in agents]
    
    async def _run(self, messages: List[Message], **kwargs) -> AsyncIterator[List[Message]]:
        """Coordinate multiple agents in group chat"""
        conversation_history = messages.copy()
        
        for round_num in range(kwargs.get('max_rounds', 5)):
            # Select next speaker
            selected_agent = await self._select_next_agent(conversation_history)
            
            # Get agent response
            agent_response = await selected_agent.run(conversation_history)
            
            # Update conversation
            conversation_history.extend(agent_response)
            yield agent_response
            
            # Check for completion
            if self._is_conversation_complete(conversation_history):
                break
    
    async def _select_next_agent(self, conversation: List[Message]) -> BaseAgent:
        """Select next agent based on strategy"""
        if self.selection_method == 'round_robin':
            return self._select_round_robin()
        elif self.selection_method == 'random':
            return self._select_random()
        elif self.selection_method == 'auto':
            return await self._select_auto(conversation)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    
    async def _select_auto(self, conversation: List[Message]) -> BaseAgent:
        """Auto-select agent using LLM"""
        # Implementation would use LLM to decide who should speak next
        prompt = self._build_selection_prompt(conversation)
        # ... LLM call logic
        pass
```

#### Step 2: Agent Communication Protocol
```python
# agents/communication.py
class AgentCommunication:
    """Handles communication between agents"""
    
    def __init__(self):
        self.message_queue = asyncio.Queue()
        self.subscribers = {}  # agent_name -> callback
    
    def subscribe(self, agent_name: str, callback):
        """Subscribe agent to messages"""
        self.subscribers[agent_name] = callback
    
    async def broadcast(self, from_agent: str, message: Message):
        """Broadcast message to all agents except sender"""
        for agent_name, callback in self.subscribers.items():
            if agent_name != from_agent:
                await callback(message)
    
    async def send_direct(self, from_agent: str, to_agent: str, message: Message):
        """Send direct message between agents"""
        if to_agent in self.subscribers:
            await self.subscribers[to_agent](message)
```

---

## 📋 Guide 4: Advanced Loop Detection

**Priority**: HIGH  
**Time**: 4-6 hours  
**Impact**: Prevents infinite loops and improves reliability

### Implementation Pattern (from qwen-code loopDetectionService.ts)

#### Step 1: Enhanced Loop Detection Class
```python
# safety/advanced_loop_detection.py
import hashlib
from typing import List, Dict, Optional, Set
from dataclasses import dataclass

@dataclass
class LoopDetectionConfig:
    tool_call_threshold: int = 5
    content_threshold: int = 10
    content_chunk_size: int = 50
    max_history_length: int = 1000
    llm_check_after_turns: int = 30

class AdvancedLoopDetector:
    def __init__(self, config: LoopDetectionConfig):
        self.config = config
        self.tool_call_history = []  # List of tool calls
        self.content_history = ""    # Aggregated content
        self.content_stats = {}      # Content frequency analysis
        self.turn_count = 0
        self.loop_detected = False
        
    def analyze_tool_call(self, tool_calls: List[Dict]) -> Optional[Dict]:
        """Detect tool call patterns"""
        if not tool_calls:
            return None
            
        # Simple repetition detection
        tool_names = [call.get('name', '') for call in tool_calls]
        tool_key = '|'.join(sorted(tool_names))
        
        if len(self.tool_call_history) > 0:
            last_tool_key = self.tool_call_history[-1]
            if tool_key == last_tool_key:
                count = getattr(self, '_tool_repetition_count', 0) + 1
                setattr(self, '_tool_repetition_count', count)
                
                if count >= self.config.tool_call_threshold:
                    return {
                        'type': 'tool_repetition',
                        'tool_calls': tool_calls,
                        'repetition_count': count,
                        'confidence': 'high'
                    }
            else:
                setattr(self, '_tool_repetition_count', 0)
        
        self.tool_call_history.append(tool_key)
        if len(self.tool_call_history) > self.config.max_history_length:
            self.tool_call_history = self.tool_call_history[-self.config.max_history_length:]
        
        return None
    
    def analyze_content(self, content: str) -> Optional[Dict]:
        """Detect content repetition patterns"""
        # Clean and normalize content
        normalized_content = self._normalize_content(content)
        
        # Analyze sentence patterns
        sentences = normalized_content.split('.')
        sentence_hashes = []
        
        for sentence in sentences:
            if sentence.strip():
                sentence_hash = hashlib.md5(sentence.strip().encode()).hexdigest()
                sentence_hashes.append(sentence_hash)
                
                # Track frequency
                if sentence_hash not in self.content_stats:
                    self.content_stats[sentence_hash] = []
                self.content_stats[sentence_hash].append(len(self.content_stats[sentence_hash]))
        
        # Check for repetition
        for hash_value, positions in self.content_stats.items():
            if len(positions) >= self.config.content_threshold:
                return {
                    'type': 'content_repetition',
                    'repetition_count': len(positions),
                    'confidence': 'high' if len(positions) >= 8 else 'medium'
                }
        
        return None
    
    def should_use_llm_detection(self) -> bool:
        """Determine if LLM-based detection should be used"""
        self.turn_count += 1
        return self.turn_count >= self.config.llm_check_after_turns
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for comparison"""
        return content.lower().strip().replace('\n', ' ').replace('  ', ' ')
```

#### Step 2: Integration with Agent System
```python
# Update agents/base_agent.py
from safety.advanced_loop_detection import AdvancedLoopDetector

class BaseAgent(ABC):
    def __init__(self, config: AgentConfig):
        # ... existing code
        self.loop_detector = AdvancedLoopDetector(LoopDetectionConfig())
    
    async def _run(self, messages: List[Message], **kwargs):
        """Enhanced run method with loop detection"""
        self.loop_detector.turn_count = 0
        
        async for response in self._run_implementation(messages, **kwargs):
            # Check for loops in response
            if response:
                tool_calls = self._extract_tool_calls(response)
                content = self._extract_content(response)
                
                # Tool call pattern analysis
                loop_detected = self.loop_detector.analyze_tool_call(tool_calls)
                if loop_detected:
                    logger.warning(f"Tool call loop detected: {loop_detected}")
                    break
                
                # Content repetition analysis
                loop_detected = self.loop_detector.analyze_content(content)
                if loop_detected:
                    logger.warning(f"Content loop detected: {loop_detected}")
                    break
            
            yield response
```

---

## 📋 Guide 5: Response Caching System

**Priority**: MEDIUM  
**Time**: 4-6 hours  
**Impact**: Significant performance improvement

### Implementation Pattern (from Qwen-Agent base.py)

#### Step 1: Cache Manager
```python
# utils/cache_manager.py
import json
import diskcache as dc
from typing import Any, Optional, Dict, List
from utils.retry import json_dumps_compact

class CacheManager:
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir:
            self.cache = dc.Cache(cache_dir)
        else:
            self.cache = None
    
    def get_cache_key(self, messages: List[Dict], functions: List[Dict], extra_cfg: Dict) -> str:
        """Generate cache key from request parameters"""
        cache_data = {
            'messages': messages,
            'functions': functions,
            'extra_cfg': extra_cfg
        }
        return json_dumps_compact(cache_data, sort_keys=True)
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached response"""
        if not self.cache:
            return None
            
        value = self.cache.get(key)
        if value:
            return json.loads(value)
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Cache response"""
        if not self.cache:
            return
            
        self.cache.set(key, json_dumps_compact(value))
    
    def invalidate(self, pattern: str = None) -> None:
        """Invalidate cache"""
        if self.cache and pattern:
            # Invalidate keys matching pattern
            for key in list(self.cache.iterkeys()):
                if pattern in key:
                    del self.cache[key]
```

#### Step 2: Integrate with Provider
```python
# provider.py - Add caching to BaseProvider
class BaseProvider(ABC):
    def __init__(self, api_key: str, model: str, **kwargs):
        # ... existing code
        cache_dir = kwargs.get('cache_dir')
        self.cache_manager = CacheManager(cache_dir)
    
    async def create_chat_completion(self, messages, tools=None, stream=False, **kwargs):
        # Check cache first (for non-streaming)
        if not stream and self.cache_manager:
            cache_key = self.cache_manager.get_cache_key(messages, tools, kwargs)
            cached_response = self.cache_manager.get(cache_key)
            if cached_response:
                return ModelResponse(**cached_response)
        
        # Make actual request
        response = await self._make_request(messages, tools, stream, **kwargs)
        
        # Cache response (for non-streaming)
        if not stream and self.cache_manager:
            cache_key = self.cache_manager.get_cache_key(messages, tools, kwargs)
            self.cache_manager.set(cache_key, response.__dict__)
        
        return response
```

---

## 📋 Guide 6: MCP Manager Singleton

**Priority**: HIGH  
**Time**: 3-4 hours  
**Impact**: Centralized MCP server management

### Implementation Pattern (from Qwen-Agent mcp_manager.py)

#### Step 1: MCP Manager Singleton
```python
# tools/mcp_manager.py
import asyncio
import threading
from typing import Dict, Optional

class MCPManager:
    _instance = None  # Private class variable for singleton
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MCPManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'servers'):  # Only initialize once
            self.servers = {}  # server_name -> server_info
            self.clients = {}  # server_name -> client_session
            self.processes = []  # Track subprocesses for cleanup
            
            # Setup async event loop
            self.loop = asyncio.new_event_loop()
            self.loop_thread = threading.Thread(target=self._start_loop, daemon=True)
            self.loop_thread.start()
    
    def _start_loop(self):
        """Start async event loop in separate thread"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    async def add_mcp_server(self, name: str, config: Dict):
        """Add MCP server configuration"""
        self.servers[name] = config
        logger.info(f"Added MCP server: {name}")
    
    async def connect_server(self, name: str) -> bool:
        """Connect to MCP server"""
        if name not in self.servers:
            raise ValueError(f"MCP server '{name}' not found")
        
        # Connect to server using MCP protocol
        # ... implementation based on server type
        pass
    
    async def get_tools(self, server_name: str) -> List[Dict]:
        """Get tools from specific server"""
        # Implementation to fetch tool list from MCP server
        pass
    
    def get_all_tools(self) -> Dict[str, List[Dict]]:
        """Get all tools from all connected servers"""
        all_tools = {}
        for server_name in self.servers.keys():
            if server_name in self.clients:
                all_tools[server_name] = self.get_tools(server_name)
        return all_tools
    
    async def health_check(self, server_name: str) -> Dict:
        """Check health of MCP server"""
        # Implementation to ping server and check status
        pass
```

#### Step 2: Global MCP Instance
```python
# tools/__init__.py - Add global MCP manager
from .mcp_manager import MCPManager

# Global MCP manager instance
mcp_manager = MCPManager()
```

---

## 📋 Guide 7: Enhanced Streaming Support

**Priority**: HIGH  
**Time**: 4-6 hours  
**Impact: Real-time user experience

### Implementation Pattern (from Qwen-Agent base.py)

#### Step 1: Streaming Response Handler
```python
# llm/streaming.py
from typing import AsyncGenerator, List, Dict, Any

class StreamingResponseHandler:
    def __init__(self, provider, messages, tools=None, **kwargs):
        self.provider = provider
        self.messages = messages
        self.tools = tools or []
        self.kwargs = kwargs
        self.accumulated_response = ""
        self.tool_calls = []
        self.metadata = {}
    
    async def stream_response(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle streaming response with real-time updates"""
        try:
            # Start streaming
            async for chunk in self.provider.create_chat_completion(
                messages=self.messages,
                tools=self.tools,
                stream=True,
                **self.kwargs
            ):
                yield self._process_chunk(chunk)
                
        except Exception as e:
            yield {
                'type': 'error',
                'error': str(e),
                'metadata': {'error_type': type(e).__name__}
            }
    
    def _process_chunk(self, chunk) -> Dict[str, Any]:
        """Process streaming chunk"""
        if hasattr(chunk, 'content') and chunk.content:
            self.accumulated_response += chunk.content
            
            return {
                'type': 'content',
                'content': chunk.content,
                'accumulated': self.accumulated_response,
                'metadata': getattr(chunk, 'metadata', {})
            }
        
        if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
            self.tool_calls.extend(chunk.tool_calls)
            
            return {
                'type': 'tool_call',
                'tool_calls': chunk.tool_calls,
                'metadata': getattr(chunk, 'metadata', {})
            }
        
        return {'type': 'heartbeat'}
```

#### Step 2: Real-time WebSocket Updates
```python
# client.py - Enhanced WebSocket with streaming
class MCPClientSystem:
    async def process_message_streaming(self, session_id: str, message: str):
        """Stream response with real-time WebSocket updates"""
        # Send initial response
        await self.websocket_manager.send_update(session_id, {
            'type': 'processing_start',
            'message': 'Starting response...'
        })
        
        try:
            # Get streaming response
            stream_handler = StreamingResponseHandler(
                provider=self.get_provider(),
                messages=[{'role': 'user', 'content': message}]
            )
            
            async for chunk in stream_handler.stream_response():
                # Send chunk to client
                await self.websocket_manager.send_update(session_id, chunk)
                
                # If tool calls are detected, execute them
                if chunk.get('type') == 'tool_call':
                    await self._execute_tool_calls(session_id, chunk['tool_calls'])
            
            # Send completion
            await self.websocket_manager.send_update(session_id, {
                'type': 'processing_complete'
            })
            
        except Exception as e:
            await self.websocket_manager.send_update(session_id, {
                'type': 'error',
                'error': str(e)
            })
```

---

## 🚀 Quick Implementation Checklist

### This Week (Start Here)
- [ ] **Tool Registry Pattern** (2-3 hours) - Guide 1
- [ ] **Agent Registry System** (3-4 hours) - Guide 2  
- [ ] **Enhanced Loop Detection** (4-6 hours) - Guide 4
- [ ] **Response Caching** (4-6 hours) - Guide 5

### Next Week
- [ ] **MCP Manager Singleton** (3-4 hours) - Guide 6
- [ ] **Multi-Agent Coordination** (6-8 hours) - Guide 3
- [ ] **Enhanced Streaming** (4-6 hours) - Guide 7

### Following Week
- [ ] **Memory Agent Implementation**
- [ ] **Group Chat System**
- [ ] **Performance Optimization**
- [ ] **Enhanced Monitoring**

---

## 💡 Implementation Tips

### Testing Strategy
1. **Unit test each component** as you build it
2. **Integration test** the registry patterns
3. **Performance test** caching and streaming
4. **Load test** multi-agent coordination

### Error Handling
1. **Graceful degradation** when features fail
2. **Comprehensive logging** for debugging
3. **User-friendly error messages**
4. **Retry mechanisms** for transient failures

### Performance Considerations
1. **Lazy loading** for tools and agents
2. **Connection pooling** for databases and APIs
3. **Memory management** for long-running sessions
4. **Cache invalidation** strategies

**Start with Guide 1 (Tool Registry) - it's the foundation for everything else!**