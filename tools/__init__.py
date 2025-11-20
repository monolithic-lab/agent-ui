# tools/__init__.py
"""
Agent-UI Tools System
Dynamic tool registry and factory pattern implementation
"""

# =============================================================================
# TOOL REGISTRY SYSTEM - Define first to avoid circular imports
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
    _auto_register_tools()


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


def _auto_register_tools():
    """Automatically register tools by scanning for tool classes"""
    # Import all tool modules
    from . import base_tool, code_interpreter
    
    # Auto-register classes that have __tool_name__ attribute
    for attr_name in dir(code_interpreter):
        attr = getattr(code_interpreter, attr_name)
        if (isinstance(attr, type) and 
            hasattr(attr, '__tool_name__') and 
            attr.__tool_name__ not in TOOL_REGISTRY):
            
            tool_name = attr.__tool_name__
            TOOL_REGISTRY[tool_name] = attr
            
            # Add registry metadata
            attr._registry_name = tool_name
            attr._registered_at = __import__('datetime').datetime.now()
            
            print(f"✅ Auto-registered tool: {tool_name}")


# =============================================================================
# BASE CLASSES
# =============================================================================

from .base_tool import BaseTool, ToolResult, ToolSchema

# Import all tool modules
from . import code_interpreter
CodeInterpreter = code_interpreter.CodeInterpreter

# Auto-register tools after all modules are loaded
_auto_register_tools()

# Print initial state
print(f"🔧 Tools module loaded. Registered tools: {list_available_tools()}")

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