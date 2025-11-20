# agents/base_agent.py
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator, AsyncIterator
from dataclasses import dataclass

from llm.base_chat_model import BaseChatModel
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
    rag_enabled: bool = False
    memory_enabled: bool = False

class BaseAgent(ABC):
    """Base class for all agents"""
    
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
            response = await self.llm.chat_with_retry(
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