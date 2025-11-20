# 🚀 Quick Start: Implement Tool Registry (Your First Win)

**Goal**: Transform your tools system to match Qwen-Agent's architecture  
**Time**: 2-3 hours  
**Impact**: Foundation for all advanced features  

---

## 📝 What We'll Build

A registry pattern that allows:
- Dynamic tool discovery: `TOOL_REGISTRY['code_interpreter']`
- Factory creation: `create_tool('code_interpreter', **config)`
- Plugin loading: Automatic registration of new tools
- Tool management: List, enable/disable, configure tools

---

## 🎯 Step-by-Step Implementation

### Step 1: Update `tools/__init__.py` (5 minutes)

Replace your current file content:

```python
# tools/__init__.py
"""
Agent-UI Tools System
Dynamic tool registry and factory pattern implementation
"""

from .base_tool import BaseTool, ToolResult, ToolSchema
from .code_interpreter import CodeInterpreter

# =============================================================================
# TOOL REGISTRY SYSTEM
# =============================================================================

# Global tool registry - mirrors Qwen-Agent pattern
TOOL_REGISTRY = {}

# Tool instances registry - for performance (singleton pattern)
_TOOL_INSTANCES = {}


def register_tool(name: str = None):
    """
    Decorator to register tools in the global registry
    
    Usage:
        @register_tool('my_tool')
        class MyTool(BaseTool):
            pass
    """
    def decorator(cls):
        tool_name = name or getattr(cls, '__tool_name__', cls.__name__.lower())
        TOOL_REGISTRY[tool_name] = cls
        
        # Add registry metadata
        cls._registry_name = tool_name
        cls._registered_at = __import__('datetime').datetime.now()
        
        print(f"✅ Registered tool: {tool_name}")
        return cls
    
    return decorator


def get_tool_registry() -> dict:
    """Get copy of the tool registry"""
    return TOOL_REGISTRY.copy()


def list_available_tools() -> list:
    """List all registered tools"""
    return list(TOOL_REGISTRY.keys())


def create_tool(tool_name: str, **kwargs):
    """
    Factory function to create tool instances
    
    Args:
        tool_name: Name of the tool to create
        **kwargs: Arguments to pass to tool constructor
    
    Returns:
        BaseTool: Tool instance
    
    Raises:
        ValueError: If tool not found in registry
    """
    if tool_name not in TOOL_REGISTRY:
        available = ', '.join(TOOL_REGISTRY.keys())
        raise ValueError(
            f"Tool '{tool_name}' not found in registry. "
            f"Available tools: {available}"
        )
    
    tool_class = TOOL_REGISTRY[tool_name]
    return tool_class(**kwargs)


def get_tool_instance(tool_name: str, **kwargs):
    """
    Get or create singleton tool instance
    
    Args:
        tool_name: Name of the tool
        **kwargs: Arguments to create tool if it doesn't exist
    
    Returns:
        BaseTool: Tool instance (singleton)
    """
    if tool_name not in _TOOL_INSTANCES:
        _TOOL_INSTANCES[tool_name] = create_tool(tool_name, **kwargs)
    
    return _TOOL_INSTANCES[tool_name]


def reload_tool_registry():
    """Reload the tool registry (useful for development)"""
    global TOOL_REGISTRY, _TOOL_INSTANCES
    TOOL_REGISTRY.clear()
    _TOOL_INSTANCES.clear()
    
    # Re-import all tool modules to trigger registration
    from . import code_interpreter  # This will trigger @register_tool decorator


def get_tool_info(tool_name: str) -> dict:
    """Get information about a registered tool"""
    if tool_name not in TOOL_REGISTRY:
        return {}
    
    tool_class = TOOL_REGISTRY[tool_name]
    
    return {
        'name': tool_name,
        'class': tool_class.__name__,
        'module': tool_class.__module__,
        'doc': tool_class.__doc__,
        'registered_at': getattr(tool_class, '_registered_at', 'Unknown')
    }

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'BaseTool',
    'ToolResult', 
    'ToolSchema',
    'CodeInterpreter',
    'register_tool',
    'get_tool_registry',
    'list_available_tools',
    'create_tool',
    'get_tool_instance',
    'reload_tool_registry',
    'get_tool_info',
    'TOOL_REGISTRY'
]
```

### Step 2: Update `tools/base_tool.py` (10 minutes)

Add registry support to your BaseTool class:

```python
# tools/base_tool.py - Add these imports and features

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from tools import register_tool, TOOL_REGISTRY

logger = logging.getLogger(__name__)

@dataclass
class ToolSchema:
    """Tool schema for function calling"""
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str]

class BaseTool(ABC):
    """Base class for all tools with registry support"""
    
    def __init_subclass__(cls, **kwargs):
        """Auto-register subclasses when they inherit from BaseTool"""
        super().__init_subclass__(**kwargs)
        
        # Auto-register subclasses that have __tool_name__ or registry decorator
        if hasattr(cls, '__tool_name__') and cls.__tool_name__ not in TOOL_REGISTRY:
            # This is handled by the decorator, but also handle direct inheritance
            register_tool(cls.__tool_name__)(cls)
    
    def __init__(self, name: str, description: str, enabled: bool = True):
        self.name = name
        self.description = description
        self.enabled = enabled
        self._execution_count = 0
        self._error_count = 0
        self._last_execution = None
        self._registry_metadata = getattr(self.__class__, '_registry_name', name)
        
        logger.info(f"Initialized tool: {self.name} (enabled: {enabled})")
    
    @abstractmethod
    async def execute(self, arguments: Dict[str, Any]) -> 'ToolResult':
        """Execute the tool with given arguments"""
        pass
    
    @abstractmethod
    def get_schema(self) -> ToolSchema:
        """Get the tool schema for function calling"""
        pass
    
    # ... keep all your existing methods unchanged ...
    
    def get_info(self) -> Dict[str, Any]:
        """Get detailed tool information"""
        return {
            'name': self.name,
            'description': self.description,
            'enabled': self.enabled,
            'execution_stats': self.execution_stats,
            'last_execution': self._last_execution,
            'registry_name': self._registry_metadata
        }
    
    @classmethod
    def from_registry(cls, name: str, **kwargs):
        """Create tool from registry"""
        if name not in TOOL_REGISTRY:
            raise ValueError(f"Tool '{name}' not found in registry")
        
        tool_class = TOOL_REGISTRY[name]
        return tool_class(**kwargs)

# Keep your existing ToolResult class unchanged
@dataclass
class ToolResult:
    # ... existing implementation ...
```

### Step 3: Update `tools/code_interpreter.py` (10 minutes)

Convert your existing code interpreter to use the registry pattern:

```python
# tools/code_interpreter.py - Add registry decorator

import asyncio
import atexit
import json
import logging
import uuid
import os
from typing import Dict, List, Optional

from tools import register_tool  # Add this import
from tools.base_tool import BaseTool, ToolResult, ToolSchema

logger = logging.getLogger(__name__)

@register_tool('code_interpreter')  # Add this decorator
class CodeInterpreter(BaseTool):
    """Python code interpreter tool with registry support"""
    
    # Registry metadata
    __tool_name__ = 'code_interpreter'  # This will auto-register
    
    def __init__(self, cfg: Optional[Dict] = None):
        cfg = cfg or {}
        super().__init__(
            name='code_interpreter',
            description='Python code sandbox, which can be used to execute Python code.',
            enabled=cfg.get('enabled', True)
        )
        
        self.work_dir = cfg.get('work_dir', '/tmp/agent_code_interpreter')
        self.instance_id = str(uuid.uuid4())
        self.timeout = cfg.get('timeout', 30)
        
        # Ensure work directory exists
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Setup cleanup on exit
        atexit.register(self._cleanup)
        
        logger.info(f"CodeInterpreter initialized: {self.instance_id}")
    
    def get_schema(self) -> ToolSchema:
        """Get tool schema for function calling"""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                'type': 'object',
                'properties': {
                    'code': {
                        'type': 'string',
                        'description': 'The python code to execute.'
                    }
                },
                'required': ['code']
            },
            required=['code']
        )
    
    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute Python code"""
        try:
            self._execution_count += 1
            self._last_execution = __import__('datetime').datetime.now()
            
            code = arguments.get('code', '')
            if not code.strip():
                return ToolResult(
                    content="Error: No code provided",
                    success=False,
                    metadata={'tool': self.name}
                )
            
            # Execute code (simplified - you'd implement actual execution)
            result = await self._execute_code(code)
            
            return ToolResult(
                content=result,
                success=True,
                metadata={
                    'tool': self.name,
                    'execution_id': self.instance_id,
                    'code_length': len(code)
                }
            )
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Code execution failed: {e}")
            return ToolResult(
                content=f"Error executing code: {str(e)}",
                success=False,
                metadata={'tool': self.name, 'error': str(e)}
            )
    
    async def _execute_code(self, code: str) -> str:
        """Execute code in sandboxed environment"""
        # Your existing code execution logic here
        # For now, just return a placeholder
        return f"Executed {len(code)} characters of code successfully"
    
    def _cleanup(self):
        """Cleanup resources"""
        try:
            # Your cleanup logic here
            pass
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    @property
    def execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        stats = super().execution_stats
        stats.update({
            'instance_id': self.instance_id,
            'work_dir': self.work_dir,
            'timeout': self.timeout
        })
        return stats
```

### Step 4: Test Your Implementation (15 minutes)

Create a test file to verify everything works:

```python
# test_tool_registry.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools import (
    TOOL_REGISTRY, 
    get_tool_registry, 
    list_available_tools,
    create_tool,
    get_tool_instance,
    get_tool_info
)

def test_tool_registry():
    """Test the new tool registry system"""
    
    print("🧪 Testing Tool Registry System")
    print("=" * 40)
    
    # 1. Check registry population
    print(f"📋 Registered tools: {list_available_tools()}")
    assert 'code_interpreter' in TOOL_REGISTRY, "CodeInterpreter should be registered"
    
    # 2. Test tool creation
    print("\n🔧 Creating tools...")
    code_interpreter = create_tool('code_interpreter', enabled=True)
    print(f"✅ Created tool: {code_interpreter.name}")
    
    # 3. Test singleton pattern
    print("\n🏠 Testing singleton pattern...")
    instance1 = get_tool_instance('code_interpreter')
    instance2 = get_tool_instance('code_interpreter')
    assert instance1 is instance2, "Should be same instance"
    print("✅ Singleton working")
    
    # 4. Test tool info
    print("\n📊 Tool information:")
    info = get_tool_info('code_interpreter')
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 5. Test tool execution
    print("\n⚡ Testing tool execution...")
    import asyncio
    
    async def test_execution():
        result = await code_interpreter.safe_execute({
            'code': 'print("Hello from registry!")'
        })
        print(f"Execution result: {result.content}")
        assert result.success, "Tool execution should succeed"
    
    asyncio.run(test_execution())
    
    # 6. Test error handling
    print("\n❌ Testing error handling...")
    try:
        create_tool('non_existent_tool')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")
    
    print("\n🎉 All tests passed!")
    print("\n🚀 Your tool registry is working! Next steps:")
    print("   1. Add more tools with @register_tool decorator")
    print("   2. Implement agent registry system")
    print("   3. Add tool discovery and loading")
    print("   4. Build tool management UI")

if __name__ == "__main__":
    test_tool_registry()
```

Run the test:
```bash
cd /workspace/agent-ui
python test_tool_registry.py
```

---

## 🎯 What You've Accomplished

✅ **Registry Pattern**: Dynamic tool registration and discovery  
✅ **Factory System**: Create tools by name with configuration  
✅ **Singleton Pattern**: Efficient tool instance management  
✅ **Metadata System**: Tool information and statistics  
✅ **Error Handling**: Proper error messages and validation  

---

## 🚀 Next Steps (This Week)

### Immediate (Next 1-2 hours)
1. **Add More Tools**: Register more tools with `@register_tool` decorator
2. **Test Thoroughly**: Run the test and fix any issues
3. **Agent Registry**: Apply same pattern to agents (see Implementation Guides)

### This Week
1. **Tool Management UI**: Build interface to manage tools
2. **Dynamic Loading**: Load tools from plugins/files
3. **Performance Optimization**: Tool pooling and caching
4. **Integration**: Connect to your existing agent system

### Next Week
1. **Agent Registry System**: Apply same pattern to agents
2. **Multi-Agent Coordination**: Group chat and agent communication
3. **Enhanced Features**: Loop detection, caching, streaming

---

## 💡 Pro Tips

### Pattern Usage
```python
# Register a new tool
@register_tool('web_search')
class WebSearchTool(BaseTool):
    __tool_name__ = 'web_search'
    
    def __init__(self, **kwargs):
        super().__init__(
            name='web_search',
            description='Search the web for information'
        )

# Use in your code
web_search = create_tool('web_search', api_key='your-key')
result = await web_search.execute({'query': 'AI trends 2025'})
```

### Debugging Tools
```python
# Check what's registered
print(f"Tools: {list_available_tools()}")
print(f"Registry: {get_tool_registry()}")

# Get tool details
for tool_name in list_available_tools():
    info = get_tool_info(tool_name)
    print(f"{tool_name}: {info}")
```

### Best Practices
1. **Always use `@register_tool` decorator** on tool classes
2. **Provide clear descriptions** for better user experience  
3. **Implement proper error handling** in tool execution
4. **Use singleton pattern** for tools with expensive initialization
5. **Test thoroughly** before moving to next feature

---

**🎉 Congratulations! You've implemented the foundation pattern that Qwen-Agent uses. This single change sets up your entire framework for the advanced features ahead!**