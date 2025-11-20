# llm/schema.py
"""
Message schemas and models for LLM interactions
Inspired by OpenAI and Anthropic patterns
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from enum import Enum

# Message roles
USER = 'user'
ASSISTANT = 'assistant' 
SYSTEM = 'system'
TOOL_RESULT = 'tool_result'
TOOL_CALL = 'tool_call'

# Content types
CHAT_COMPLETION = 'chat.completion'
CHAT_MESSAGE = 'chat.message'

@dataclass
class FunctionCall:
    """Function call in OpenAI format"""
    id: str
    name: str
    arguments: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': 'function',
            'function': {
                'name': self.name,
                'arguments': self.arguments
            }
        }

@dataclass 
class ToolCall:
    """Tool call in Claude format"""
    id: str
    name: str
    input: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': 'tool_use',
            'name': self.name,
            'input': self.input
        }

@dataclass
class Message:
    """Standardized message format"""
    role: str
    content: str
    thinking_content: Optional[str] = None
    tool_calls: Optional[List[Union[FunctionCall, ToolCall]]] = None
    tool_call_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    name: Optional[str] = None  # For function calls
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI message format"""
        message_data = {
            'role': self.role,
            'content': self.content or ''
        }
        
        # Handle tool calls
        if self.role == ASSISTANT and self.tool_calls:
            tool_calls_data = []
            for tool_call in self.tool_calls:
                if isinstance(tool_call, FunctionCall):
                    tool_calls_data.append(tool_call.to_dict())
                elif isinstance(tool_call, ToolCall):
                    tool_calls_data.append({
                        'id': tool_call.id,
                        'type': 'function',
                        'function': {
                            'name': tool_call.name,
                            'arguments': str(tool_call.input)
                        }
                    })
            message_data['tool_calls'] = tool_calls_data
        
        # Handle tool result
        if self.role == TOOL_RESULT and self.tool_call_id:
            message_data['tool_call_id'] = self.tool_call_id
        
        return message_data
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic message format"""
        message_data = {
            'role': self.role,
            'content': self.content or ''
        }
        
        # Handle tool calls
        if self.role == ASSISTANT and self.tool_calls:
            content_blocks = [{'type': 'text', 'text': self.content or ''}]
            
            for tool_call in self.tool_calls:
                if isinstance(tool_call, ToolCall):
                    content_blocks.append({
                        'type': 'tool_use',
                        'id': tool_call.id,
                        'name': tool_call.name,
                        'input': tool_call.input
                    })
            
            message_data['content'] = content_blocks
        
        # Handle tool result
        if self.role == TOOL_RESULT:
            content_blocks = [{
                'type': 'tool_result',
                'tool_use_id': self.tool_call_id,
                'content': self.content or ''
            }]
            message_data['content'] = content_blocks
        
        return message_data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'role': self.role,
            'content': self.content,
            'thinking_content': self.thinking_content,
            'tool_calls': [tc.to_dict() if hasattr(tc, 'to_dict') else tc for tc in self.tool_calls] if self.tool_calls else None,
            'tool_call_id': self.tool_call_id,
            'metadata': self.metadata,
            'name': self.name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        return cls(
            role=data['role'],
            content=data.get('content', ''),
            thinking_content=data.get('thinking_content'),
            tool_calls=data.get('tool_calls'),
            tool_call_id=data.get('tool_call_id'),
            metadata=data.get('metadata'),
            name=data.get('name')
        )