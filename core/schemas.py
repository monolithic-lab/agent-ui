# core/schemas.py
from dataclasses import dataclass
from enum import Enum
from typing import List, Literal, Optional, Tuple, Union, Dict, Any
from pydantic import BaseModel, field_validator, model_validator

# Message roles
USER = 'user'
ASSISTANT = 'assistant' 
SYSTEM = 'system'
TOOL_RESULT = 'tool_result'
TOOL_CALL = 'tool_call'

# Content types
CHAT_COMPLETION = 'chat.completion'
CHAT_MESSAGE = 'chat.message'

DEFAULT_SYSTEM_MESSAGE = ''

ROLE = 'role'
CONTENT = 'content'
REASONING_CONTENT = 'reasoning_content'
NAME = 'name'

SYSTEM = 'system'
USER = 'user'
ASSISTANT = 'assistant'
FUNCTION = 'function'

FILE = 'file'
IMAGE = 'image'
AUDIO = 'audio'
VIDEO = 'video'

class PathConfig(BaseModel):
    work_space_root: str
    download_root: str
    code_interpreter_ws: str


class ServerConfig(BaseModel):
    server_host: str
    fast_api_port: int
    app_in_browser_port: int
    workstation_port: int
    model_server: str
    api_key: str
    llm: str
    max_ref_token: int
    max_days: int

    class Config:
        protected_namespaces = ()


class GlobalConfig(BaseModel):
    path: PathConfig
    server: ServerConfig
    
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

class BaseModelCompatibleDict(BaseModel):

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def model_dump(self, **kwargs):
        if 'exclude_none' not in kwargs:
            kwargs['exclude_none'] = True
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs):
        if 'exclude_none' not in kwargs:
            kwargs['exclude_none'] = True
        return super().model_dump_json(**kwargs)

    def get(self, key, default=None):
        try:
            value = getattr(self, key)
            if value:
                return value
            else:
                return default
        except AttributeError:
            return default

    def __str__(self):
        return f'{self.model_dump()}'


class FunctionCall(BaseModelCompatibleDict):
    name: str
    arguments: str

    def __init__(self, name: str, arguments: str):
        super().__init__(name=name, arguments=arguments)

    def __repr__(self):
        return f'FunctionCall({self.model_dump()})'


class ContentItem(BaseModelCompatibleDict):
    text: Optional[str] = None
    image: Optional[str] = None
    file: Optional[str] = None
    audio: Optional[Union[str, dict]] = None
    video: Optional[Union[str, list]] = None

    def __init__(self,
                 text: Optional[str] = None,
                 image: Optional[str] = None,
                 file: Optional[str] = None,
                 audio: Optional[Union[str, dict]] = None,
                 video: Optional[Union[str, list]] = None):
        super().__init__(text=text, image=image, file=file, audio=audio, video=video)

    @model_validator(mode='after')
    def check_exclusivity(self):
        provided_fields = 0
        if self.text is not None:
            provided_fields += 1
        if self.image:
            provided_fields += 1
        if self.file:
            provided_fields += 1
        if self.audio:
            provided_fields += 1
        if self.video:
            provided_fields += 1

        if provided_fields != 1:
            raise ValueError("Exactly one of 'text', 'image', 'file', 'audio', or 'video' must be provided.")
        return self

    def __repr__(self):
        return f'ContentItem({self.model_dump()})'

    def get_type_and_value(self) -> Tuple[Literal['text', 'image', 'file', 'audio', 'video'], str]:
        (t, v), = self.model_dump().items()
        assert t in ('text', 'image', 'file', 'audio', 'video')
        return t, v

    @property
    def type(self) -> Literal['text', 'image', 'file', 'audio', 'video']:
        t, v = self.get_type_and_value()
        return t

    @property
    def value(self) -> str:
        t, v = self.get_type_and_value()
        return v


class Message(BaseModelCompatibleDict):
    role: str
    content: Union[str, List[ContentItem]]
    reasoning_content: Optional[Union[str, List[ContentItem]]] = None
    name: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    extra: Optional[dict] = None

    def __init__(self,
                 role: str,
                 content: Union[str, List[ContentItem]],
                 reasoning_content: Optional[Union[str, List[ContentItem]]] = None,
                 name: Optional[str] = None,
                 function_call: Optional[FunctionCall] = None,
                 extra: Optional[dict] = None,
                 **kwargs):
        if content is None:
            content = ''
        super().__init__(role=role,
                         content=content,
                         reasoning_content=reasoning_content,
                         name=name,
                         function_call=function_call,
                         extra=extra)

    def __repr__(self):
        return f'Message({self.model_dump()})'

    @field_validator('role')
    def role_checker(cls, value: str) -> str:
        if value not in [USER, ASSISTANT, SYSTEM, FUNCTION]:
            raise ValueError(f'{value} must be one of {",".join([USER, ASSISTANT, SYSTEM, FUNCTION])}')
        return value
