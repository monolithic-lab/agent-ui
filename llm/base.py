# llm/base.py
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, AsyncIterator
from asyncio import StreamReader

from core.schemas import Message, ASSISTANT, ToolCall, FunctionCall
from core.provider import BaseProvider, ModelResponse
from utils.performance import monitor_performance

logger = logging.getLogger(__name__)

class BaseChatModel(ABC):
    """Base chat model interface"""
    
    def __init__(self, provider: BaseProvider):
        self.provider = provider
        self.provider_info = provider.provider_info
    
    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> List[Message]:
        """Create chat completion"""
        # Convert messages to provider format
        provider_messages = [msg.to_openai_format() for msg in messages]
        
        # Call provider
        response = await self.provider.create_chat_completion(
            messages=provider_messages,
            tools=tools,
            **kwargs
        )
        
        # Convert response to message format
        if isinstance(response, list):
            # Streaming response
            return []
        else:
            # Single response
            return self._convert_response_to_messages(response)
    
    async def chat_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> AsyncIterator[List[Message]]:
        """Create streaming chat completion"""
        # Convert messages to provider format
        provider_messages = [msg.to_openai_format() for msg in messages]
        
        # Call provider with streaming
        async for response in self.provider.create_chat_completion(
            messages=provider_messages,
            tools=tools,
            stream=True,
            **kwargs
        ):
            messages = self._convert_response_to_messages(response)
            yield messages
    
    @monitor_performance('llm.chat')
    async def chat_with_retry(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> List[Message]:
        """Create chat completion with retry"""
        from utils.retry import retry_with_backoff_async
        
        async def _chat():
            return await self.chat(messages, tools, **kwargs)
        
        return await retry_with_backoff_async(_chat)
    
    def _convert_response_to_messages(self, response: ModelResponse) -> List[Message]:
        """Convert ModelResponse to Message list"""
        messages = []
        
        # Handle tool calls
        if response.tool_calls:
            # Create assistant message with tool calls
            tool_calls = []
            for tool_call_data in response.tool_calls:
                if isinstance(tool_call_data, dict):
                    # OpenAI format
                    tool_calls.append(FunctionCall(
                        id=tool_call_data.get('id', ''),
                        name=tool_call_data['function']['name'],
                        arguments=tool_call_data['function']['arguments']
                    ))
                else:
                    # Already a FunctionCall object
                    tool_calls.append(tool_call_data)
            
            messages.append(Message(
                role=ASSISTANT,
                content=response.content or '',
                tool_calls=tool_calls,
                metadata=response.metadata
            ))
        else:
            # Regular assistant message
            messages.append(Message(
                role=ASSISTANT,
                content=response.content,
                thinking_content=response.thinking_content,
                metadata=response.metadata
            ))
        
        return messages
    
    def get_model_name(self) -> str:
        """Get model name"""
        return self.provider.model
    
    def get_provider_info(self) -> Any:
        """Get provider information"""
        return self.provider_info