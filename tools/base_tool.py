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
    """Base class for all tools"""
    
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
    
    def to_message(self, tool_call_id: str) -> 'Message':
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