# agents/base.py
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator, AsyncIterator
from dataclasses import dataclass

from llm.base import BaseChatModel
from tools.base import BaseTool
from core.exceptions import ModelServiceError
from core.schemas import Message

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
    """Base class for all agents with registry support"""
    
    # Registry metadata - override in subclasses
    __agent_name__ = 'base_agent'
    
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
    
    def get_info(self) -> Dict[str, Any]:
        """Get detailed agent information"""
        return {
            'name': self.config.name,
            'description': self.config.description,
            'agent_type': self.__agent_name__,
            'tools_count': len(self.tools),
            'tools_enabled': len([tool for tool in self.tools if tool.enabled]),
            'iterations': self._iteration_count,
            'registry_name': getattr(self.__class__, '_registry_name', self.__agent_name__)
        }
    
    @classmethod
    def from_registry(cls, agent_name: str, **kwargs):
        """Create agent from registry"""
        from .registry import AGENT_REGISTRY
        
        if agent_name not in AGENT_REGISTRY:
            raise ValueError(f"Agent '{agent_name}' not found in registry")
        
        agent_class = AGENT_REGISTRY[agent_name]
        return agent_class(**kwargs)
    
    @property
    def execution_stats(self) -> Dict[str, Any]:
        """Get agent execution statistics"""
        return {
            'name': self.config.name,
            'agent_type': self.__agent_name__,
            'total_iterations': self._iteration_count,
            'tools_count': len(self.tools),
            'tools_enabled': len([tool for tool in self.tools if tool.enabled]),
            'llm_configured': self.llm is not None
        }


# Function calling agent implementation
import json
from .base import BaseAgent, AgentConfig
from core.exceptions import ModelServiceError
from core.schemas import ASSISTANT, Message, FunctionCall, ToolCall
from tools.base import BaseTool, ToolResult

class FnCallAgent(BaseAgent):
    """Function calling agent with registry support"""
    
    # Registry metadata
    __agent_name__ = 'fncall_agent'
    
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
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    tool_results = await self._execute_function_calls(
                        response.tool_calls, 
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
        function_calls: List[Dict[str, Any]], 
        context_messages: List[Message]
    ) -> List[Message]:
        """Execute function calls and return results"""
        results = []
        
        for function_call_data in function_calls:
            try:
                # Handle different function call formats
                function_call = self._normalize_function_call(function_call_data)
                
                # Find the tool
                tool = self._find_tool(function_call.name)
                if not tool:
                    results.append(self._create_error_result(
                        function_call, f"Tool '{function_call.name}' not found"
                    ))
                    continue
                
                # Execute tool
                arguments = self._parse_arguments(function_call.arguments, tool)
                tool_result = await tool.safe_execute(arguments)
                
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
                    function_call if 'function_call' in locals() else None,
                    f"Tool execution failed: {str(e)}"
                ))
        
        return results
    
    def _normalize_function_call(self, function_call_data: Dict[str, Any]) -> FunctionCall:
        """Normalize function call to standard format"""
        if isinstance(function_call_data, dict):
            if 'function' in function_call_data:
                # OpenAI format
                return FunctionCall(
                    id=function_call_data['id'],
                    name=function_call_data['function']['name'],
                    arguments=function_call_data['function']['arguments']
                )
            elif 'input' in function_call_data:
                # Anthropic format
                return FunctionCall(
                    id=function_call_data['id'],
                    name=function_call_data['name'],
                    arguments=str(function_call_data['input'])
                )
        
        # Already in FunctionCall format
        return function_call_data
    
    def _find_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Find tool by name"""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None
    
    def _parse_arguments(self, arguments: str, tool: BaseTool) -> Dict[str, Any]:
        """Parse tool arguments safely"""
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except json.JSONDecodeError:
                raise ModelServiceError(message=f"Invalid JSON arguments for tool {tool.name}")
        else:
            return arguments if isinstance(arguments, dict) else {}
    
    def _create_error_result(self, function_call: Optional[FunctionCall], error: str) -> Message:
        """Create error result message"""
        tool_call_id = function_call.id if function_call else 'unknown'
        tool_name = function_call.name if function_call else 'unknown'
        
        return Message(
            role='tool_result',
            content=f"Error: {error}",
            tool_call_id=tool_call_id,
            metadata={'tool_name': tool_name, 'error': error}
        )