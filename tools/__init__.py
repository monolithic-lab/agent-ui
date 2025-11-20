# tools/__init__.py
"""
Agent-UI Tools System
Dynamic tool registry and factory pattern implementation
"""

# =============================================================================
# BASE CLASSES
# =============================================================================

from .base import BaseTool, ToolResult, ToolSchema

# Import all tool modules
from . import code_interpreter
CodeInterpreter = code_interpreter.CodeInterpreter

# =============================================================================
# REGISTRY FUNCTIONALITY - Import from registry module
# =============================================================================

from .registry import (
    TOOL_REGISTRY as _TOOL_REGISTRY,
    register_tool,
    get_tool_registry,
    list_available_tools,
    create_tool,
    get_tool_instance,
    reload_tool_registry,
    get_tool_info
)

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
