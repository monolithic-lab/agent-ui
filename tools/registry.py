# tools/registry.py
"""
Tool Registry System
Dynamic tool registration and factory pattern implementation
"""

import datetime
from typing import Dict, Type, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

# Global tool registry
TOOL_REGISTRY = {}

# Tool instances registry - for performance (singleton pattern)
_TOOL_INSTANCES: Dict[str, Any] = {}


def register_tool(name: Optional[str] = None):
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
        cls._registered_at = datetime.datetime.now()
        
        logger.debug(f"Registered tool: {tool_name}")
        return cls
    
    return decorator


def get_tool_registry() -> Dict[str, Type]:
    """Get copy of the tool registry"""
    return TOOL_REGISTRY.copy()


def list_available_tools() -> List[str]:
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


def get_tool_info(tool_name: str) -> Dict[str, Any]:
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
    from . import code_interpreter
    
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
            attr._registered_at = datetime.datetime.now()
            
            logger.debug(f"Auto-registered tool: {tool_name}")


# Auto-register tools on import
_auto_register_tools()

logger.debug(f"Tool registry loaded. Registered tools: {list_available_tools()}")
