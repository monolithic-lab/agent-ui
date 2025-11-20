# agents/__init__.py
"""
Agent-UI Agent System
Dynamic agent registry and factory pattern implementation
"""

# =============================================================================
# AGENT REGISTRY SYSTEM
# =============================================================================

# Global agent registry
AGENT_REGISTRY = {}

# Agent instances registry for singleton pattern
_AGENT_INSTANCES = {}


def register_agent(name: str = None):
    """
    Decorator to register agents in the global registry
    
    Usage:
        @register_agent('my_agent')
        class MyAgent(BaseAgent):
            pass
    """
    def decorator(cls):
        agent_name = name or getattr(cls, '__agent_name__', cls.__name__.lower())
        AGENT_REGISTRY[agent_name] = cls
        
        # Add registry metadata
        cls._registry_name = agent_name
        cls._registered_at = __import__('datetime').datetime.now()
        
        print(f"✅ Registered agent: {agent_name}")
        return cls
    
    return decorator


def get_agent_registry() -> dict:
    """Get copy of the agent registry"""
    return AGENT_REGISTRY.copy()


def list_available_agents() -> list:
    """List all registered agents"""
    return list(AGENT_REGISTRY.keys())


def create_agent(agent_name: str, **kwargs):
    """
    Factory function to create agent instances
    
    Args:
        agent_name: Name of the agent to create
        **kwargs: Arguments to pass to agent constructor
    
    Returns:
        BaseAgent: Agent instance
    
    Raises:
        ValueError: If agent not found in registry
    """
    if agent_name not in AGENT_REGISTRY:
        available = ', '.join(AGENT_REGISTRY.keys())
        raise ValueError(
            f"Agent '{agent_name}' not found in registry. "
            f"Available agents: {available}"
        )
    
    agent_class = AGENT_REGISTRY[agent_name]
    return agent_class(**kwargs)


def get_agent_instance(agent_name: str, **kwargs):
    """
    Get or create singleton agent instance
    
    Args:
        agent_name: Name of the agent
        **kwargs: Arguments to create agent if it doesn't exist
    
    Returns:
        BaseAgent: Agent instance (singleton)
    """
    if agent_name not in _AGENT_INSTANCES:
        _AGENT_INSTANCES[agent_name] = create_agent(agent_name, **kwargs)
    
    return _AGENT_INSTANCES[agent_name]


def reload_agent_registry():
    """Reload the agent registry (useful for development)"""
    global AGENT_REGISTRY, _AGENT_INSTANCES
    AGENT_REGISTRY.clear()
    _AGENT_INSTANCES.clear()
    
    # Re-import all agent modules to trigger registration
    _auto_register_agents()


def get_agent_info(agent_name: str) -> dict:
    """Get information about a registered agent"""
    if agent_name not in AGENT_REGISTRY:
        return {}
    
    agent_class = AGENT_REGISTRY[agent_name]
    
    return {
        'name': agent_name,
        'class': agent_class.__name__,
        'module': agent_class.__module__,
        'doc': agent_class.__doc__,
        'registered_at': getattr(agent_class, '_registered_at', 'Unknown')
    }


def _auto_register_agents():
    """Automatically register agents by scanning for agent classes"""
    # Import all agent modules
    from . import base_agent, fncall_agent, assistant
    
    # Auto-register classes that have __agent_name__ attribute
    for module in [base_agent, fncall_agent, assistant]:
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                hasattr(attr, '__agent_name__') and 
                attr.__agent_name__ not in AGENT_REGISTRY):
                
                agent_name = attr.__agent_name__
                AGENT_REGISTRY[agent_name] = attr
                
                # Add registry metadata
                attr._registry_name = agent_name
                attr._registered_at = __import__('datetime').datetime.now()
                
                print(f"✅ Auto-registered agent: {agent_name}")


# =============================================================================
# BASE CLASSES AND IMPORTS
# =============================================================================

from .base_agent import BaseAgent, AgentConfig
from .fncall_agent import FnCallAgent
from .assistant import Assistant

# Auto-register agents after all modules are loaded
_auto_register_agents()

# Print initial state
print(f"🤖 Agents module loaded. Registered agents: {list_available_agents()}")

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