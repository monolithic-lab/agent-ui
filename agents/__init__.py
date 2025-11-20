# agents/__init__.py
"""
Agent-UI Agent System
Dynamic agent registry and factory pattern implementation
"""

# =============================================================================
# BASE CLASSES AND IMPORTS
# =============================================================================

from .base import BaseAgent, AgentConfig, FnCallAgent
from .assistant import Assistant

# =============================================================================
# REGISTRY FUNCTIONALITY - Import from registry module
# =============================================================================

from .registry import (
    AGENT_REGISTRY as _AGENT_REGISTRY,
    register_agent,
    get_agent_registry,
    list_available_agents,
    create_agent,
    get_agent_instance,
    reload_agent_registry,
    get_agent_info
)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'BaseAgent',
    'AgentConfig',
    'FnCallAgent', 
    'Assistant',
    'register_agent',
    'get_agent_registry',
    'list_available_agents',
    'create_agent',
    'get_agent_instance',
    'reload_agent_registry',
    'get_agent_info',
    'AGENT_REGISTRY'
]
