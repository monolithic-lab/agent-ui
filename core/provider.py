# core/provider.py
import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass

from core.exceptions import ModelServiceError, ProviderError
from utils.retry import retry_with_backoff_async

logger = logging.getLogger(__name__)

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ModelResponse:
    """Standardized model response"""
    content: str
    thinking_content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    raw_response: Any = None
    usage: Optional[Dict] = None
    metadata: Optional[Dict] = None

@dataclass
class ProviderInfo:
    """Provider information and capabilities"""
    name: str
    supports_streaming: bool = False
    supports_thinking: bool = False
    supports_tools: bool = True
    supports_multimodal: bool = False
    models: Optional[List[str]] = None

# =============================================================================
# BASE PROVIDER CLASS
# =============================================================================

class BaseProvider(ABC):
    """Base class for all model providers"""
    
    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        self.config = kwargs
        self.provider_info = self._get_provider_info()
        
    @abstractmethod
    async def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[ModelResponse, AsyncGenerator[ModelResponse, None]]:
        """Create chat completion"""
        pass
    
    @abstractmethod
    def _format_tools(self, tools: List[Dict]) -> Any:
        """Format tools for this provider's API"""
        pass
    
    @abstractmethod
    def _parse_response(self, response: Any) -> ModelResponse:
        """Parse provider-specific response"""
        pass
    
    @abstractmethod
    def _get_provider_info(self) -> ProviderInfo:
        """Get provider capabilities and info"""
        pass
    
    def _extract_thinking(self, response: Any) -> Optional[str]:
        """Extract thinking/reasoning content from response"""
        return None
    
    def _get_usage_stats(self, response: Any) -> Optional[Dict[str, int]]:
        """Extract usage statistics from response"""
        return None
    
    async def create_chat_completion_with_retry(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[ModelResponse, AsyncGenerator[ModelResponse, None]]:
        """Create chat completion with automatic retry"""
        
        async def _make_request():
            return await self.create_chat_completion(messages, tools, stream, **kwargs)
        
        return await retry_with_backoff_async(_make_request, max_retries=3)
    
    def _handle_provider_error(self, error: Exception) -> ModelServiceError:
        """Convert provider-specific errors to ModelServiceError"""
        if isinstance(error, Exception):
            # Extract error information from different providers
            if hasattr(error, 'status_code'):
                code = str(error.status_code)
            elif hasattr(error, 'code'):
                code = error.code
            else:
                code = '500'  # Default server error
                
            message = str(error)
            return ModelServiceError(exception=error, code=code, message=message)
        
        return ModelServiceError(exception=error, code='500', message='Unknown error')

# =============================================================================
# OPENAI PROVIDER
# =============================================================================

class OpenAIProvider(BaseProvider):
    """OpenAI API provider with streaming and tool support"""
    
    def __init__(self, api_key: str, model: str = "gpt-4", base_url: str = None, **kwargs):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.base_url = base_url
        
        super().__init__(api_key, model, **kwargs)
    
    async def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[ModelResponse, AsyncGenerator[ModelResponse, None]]:
        """Create OpenAI chat completion"""
        config = {**self.config, **kwargs}
        provider_tools = self._format_tools(tools) if tools else None
        
        if stream:
            return self._stream_completion(messages, provider_tools, config)
        else:
            return await self._single_completion(messages, provider_tools, config)
    
    async def _single_completion(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict]], 
        config: Dict
    ) -> ModelResponse:
        """Single completion (non-streaming)"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                stream=False,
                **config
            )
            return self._parse_response(response)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def _stream_completion(
        self,
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict]], 
        config: Dict
    ) -> AsyncGenerator[ModelResponse, None]:
        """Streaming completion"""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                stream=True,
                **config
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield ModelResponse(
                        content=chunk.choices[0].delta.content,
                        raw_response=chunk,
                        metadata={"chunk_index": getattr(chunk, 'index', 0)}
                    )
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise
    
    def _format_tools(self, tools: List[Dict]) -> List[Dict]:
        """Format tools for OpenAI"""
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["inputSchema"]
                }
            }
            for tool in tools
        ]
    
    def _parse_response(self, response) -> ModelResponse:
        """Parse OpenAI response"""
        message = response.choices[0].message
        
        # Extract tool calls
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function", 
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
        
        # Extract thinking content (if available)
        thinking_content = self._extract_thinking(response)
        
        # Extract usage statistics
        usage = self._get_usage_stats(response)
        
        return ModelResponse(
            content=message.content or "",
            thinking_content=thinking_content,
            tool_calls=tool_calls,
            raw_response=response,
            usage=usage,
            metadata={
                "model": self.model,
                "provider": "openai",
                "base_url": self.base_url
            }
        )
    
    def _extract_thinking(self, response) -> Optional[str]:
        """Extract thinking content from OpenAI response"""
        # Check for reasoning details (newer models)
        message = response.choices[0].message
        if hasattr(message, 'reasoning_details') and message.reasoning_details:
            return message.reasoning_details[0].get('text', '')
        
        # Check for thinking blocks in content
        if hasattr(message, 'content') and isinstance(message.content, list):
            for block in message.content:
                if hasattr(block, 'type') and block.type == 'thinking':
                    return getattr(block, 'thinking', getattr(block, 'text', ''))
        
        return None
    
    def _get_usage_stats(self, response) -> Optional[Dict[str, int]]:
        """Extract usage statistics from OpenAI response"""
        if hasattr(response, 'usage'):
            return {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        return None
    
    def _get_provider_info(self) -> ProviderInfo:
        """Get OpenAI provider information"""
        return ProviderInfo(
            name="openai",
            supports_streaming=True,
            supports_thinking=True,
            supports_tools=True,
            supports_multimodal="vision" in self.model.lower(),
            models=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4-vision"]
        )

# =============================================================================
# ANTHROPIC PROVIDER
# =============================================================================

class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229", **kwargs):
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
        
        super().__init__(api_key, model, **kwargs)
    
    async def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[ModelResponse, AsyncGenerator[ModelResponse, None]]:
        """Create Anthropic chat completion"""
        config = {**self.config, **kwargs}
        anthropic_messages, system_message = self._convert_messages(messages)
        provider_tools = self._format_tools(tools) if tools else None
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                system=system_message,
                messages=anthropic_messages,
                tools=provider_tools,
                stream=stream,
                **config
            )
            
            if stream:
                return self._stream_response(response)
            else:
                return self._parse_response(response)
                
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    async def _stream_response(self, response_stream) -> AsyncGenerator[ModelResponse, None]:
        """Handle streaming response from Anthropic"""
        async for chunk in response_stream:
            if chunk.type == "content_block_delta":
                if chunk.delta.type == "text_delta":
                    yield ModelResponse(
                        content=chunk.delta.text,
                        raw_response=chunk
                    )
    
    def _convert_messages(self, messages: List[Dict[str, Any]]) -> tuple:
        """Convert OpenAI-style messages to Anthropic format"""
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                content = msg["content"]
                if isinstance(content, list):
                    anthropic_content = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                anthropic_content.append({
                                    "type": "text",
                                    "text": item.get("text", "")
                                })
                            # Handle other content types
                        else:
                            anthropic_content.append({
                                "type": "text",
                                "text": str(item)
                            })
                    content = anthropic_content
                else:
                    content = str(content)
                
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": content
                })
        
        return anthropic_messages, system_message
    
    def _format_tools(self, tools: List[Dict]) -> List[Dict]:
        """Format tools for Anthropic"""
        if not tools:
            return None
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["inputSchema"]
            }
            for tool in tools
        ]
    
    def _parse_response(self, response) -> ModelResponse:
        """Parse Anthropic response"""
        thinking_content = None
        text_content = ""
        
        for block in response.content:
            if block.type == "thinking":
                thinking_content = block.thinking
            elif block.type == "text":
                text_content += block.text
        
        # Extract tool calls if present
        tool_calls = None
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.input)
                    }
                }
                for tc in response.tool_calls
            ]
        
        # Extract usage
        usage = None
        if hasattr(response, 'usage'):
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        
        return ModelResponse(
            content=text_content,
            thinking_content=thinking_content,
            tool_calls=tool_calls,
            raw_response=response,
            usage=usage,
            metadata={
                "model": self.model,
                "provider": "anthropic",
                "stop_reason": getattr(response, 'stop_reason', None)
            }
        )
    
    def _get_provider_info(self) -> ProviderInfo:
        """Get Anthropic provider information"""
        return ProviderInfo(
            name="anthropic",
            supports_streaming=True,
            supports_thinking=True,
            supports_tools=True,
            supports_multimodal="claude-3" in self.model,
            models=["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
        )

# =============================================================================
# GOOGLE GEMINI PROVIDER
# =============================================================================

class GeminiProvider(BaseProvider):
    """Google Gemini API provider"""
    
    def __init__(self, api_key: str, model: str = "gemini-pro", **kwargs):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.genai = genai
        except ImportError:
            raise ImportError("Google GenerativeAI package not installed. Install with: pip install google-generativeai")
        
        super().__init__(api_key, model, **kwargs)
    
    async def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[ModelResponse, AsyncGenerator[ModelResponse, None]]:
        """Create Gemini chat completion"""
        config = {**self.config, **kwargs}
        gemini_messages = self._convert_messages(messages)
        
        # Configure model with tools if available
        generation_config = self.genai.types.GenerationConfig(**config)
        model = self.genai.GenerativeModel(
            model_name=self.model,
            generation_config=generation_config,
            tools=self._format_tools(tools) if tools else None
        )
        
        try:
            if len(gemini_messages) > 1:
                # Multi-turn conversation
                chat = model.start_chat(history=gemini_messages[:-1])
                response = await chat.send_message_async(gemini_messages[-1], stream=stream)
            else:
                # Single message
                response = await model.generate_content_async(gemini_messages[0], stream=stream)
            
            if stream:
                return self._stream_response(response)
            else:
                return self._parse_response(response)
                
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    async def _stream_response(self, response_stream) -> AsyncGenerator[ModelResponse, None]:
        """Handle streaming response from Gemini"""
        async for chunk in response_stream:
            if chunk.text:
                yield ModelResponse(
                    content=chunk.text,
                    raw_response=chunk
                )
    
    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Any]:
        """Convert OpenAI-style messages to Gemini format"""
        gemini_messages = []
        
        for msg in messages:
            role = "user" if msg["role"] in ["user", "system"] else "model"
            content = msg["content"]
            
            # Handle tool calls and results
            if msg["role"] == "assistant" and "tool_calls" in msg:
                # Convert tool calls to Gemini function calls
                for tool_call in msg["tool_calls"]:
                    function_call = {
                        "function_call": {
                            "name": tool_call["function"]["name"],
                            "args": json.loads(tool_call["function"]["arguments"])
                        }
                    }
                    gemini_messages.append({
                        "role": "model",
                        "parts": [function_call]
                    })
            elif msg["role"] == "tool":
                # Convert tool results
                function_response = {
                    "function_response": {
                        "name": "tool_result",
                        "response": {"content": msg["content"]}
                    }
                }
                gemini_messages.append({
                    "role": "user",
                    "parts": [function_response]
                })
            else:
                # Regular text message
                gemini_messages.append({
                    "role": role,
                    "parts": [{"text": content}]
                })
        
        return gemini_messages
    
    def _format_tools(self, tools: List[Dict]) -> List[Dict]:
        """Format tools for Gemini"""
        if not tools:
            return None
        return [
            {
                "function_declarations": [
                    {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["inputSchema"]
                    }
                ]
            }
            for tool in tools
        ]
    
    def _parse_response(self, response) -> ModelResponse:
        """Parse Gemini response"""
        content = response.text or ""
        
        # Extract function calls
        tool_calls = None
        if response.candidates and response.candidates[0].content:
            tool_calls = []
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call'):
                    tool_calls.append({
                        "id": f"call_{len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": part.function_call.name,
                            "arguments": json.dumps(part.function_call.args)
                        }
                    })
        
        # Extract usage metadata
        metadata = {
            "model": self.model,
            "provider": "gemini",
            "candidate_count": len(response.candidates) if response.candidates else 0
        }
        
        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            raw_response=response,
            metadata=metadata
        )
    
    def _get_provider_info(self) -> ProviderInfo:
        """Get Gemini provider information"""
        return ProviderInfo(
            name="gemini",
            supports_streaming=True,
            supports_thinking=False,
            supports_tools=True,
            supports_multimodal="vision" in self.model.lower(),
            models=["gemini-pro", "gemini-pro-vision"]
        )

# =============================================================================
# HUGGINGFACE PROVIDER
# =============================================================================

class HuggingFaceProvider(BaseProvider):
    """HuggingFace Inference API provider"""
    
    def __init__(self, api_key: str, model: str = None, base_url: str = None, **kwargs):
        try:
            from huggingface_hub import AsyncInferenceClient
            self.client = AsyncInferenceClient(
                api_key=api_key,
                base_url=base_url
            )
        except ImportError:
            raise ImportError("HuggingFace Hub package not installed. Install with: pip install huggingface-hub")
        
        self.base_url = base_url
        super().__init__(api_key, model, **kwargs)
    
    async def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[ModelResponse, AsyncGenerator[ModelResponse, None]]:
        """Create HuggingFace chat completion"""
        config = {**self.config, **kwargs}
        formatted_tools = self._format_tools(tools) if tools else None
        
        try:
            if stream:
                # Note: HuggingFace streaming support depends on the model
                return self._stream_completion(messages, formatted_tools, config)
            else:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=formatted_tools,
                        **config
                    )
                )
                return self._parse_response(response)
                
        except Exception as e:
            logger.error(f"HuggingFace API error: {e}")
            raise
    
    async def _stream_completion(
        self,
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict]], 
        config: Dict
    ) -> AsyncGenerator[ModelResponse, None]:
        """Handle streaming completion (model-dependent)"""
        # HuggingFace streaming depends on the specific model
        # This is a placeholder that may not work for all models
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    stream=True,
                    **config
                )
            )
            
            async for chunk in response:
                if hasattr(chunk, 'choices') and chunk.choices:
                    content = chunk.choices[0].delta.content or ""
                    if content:
                        yield ModelResponse(
                            content=content,
                            raw_response=chunk
                        )
        except Exception as e:
            logger.error(f"HuggingFace streaming error: {e}")
            # Fallback to single completion
            single_response = await self._single_completion(messages, tools, config)
            yield single_response
    
    async def _single_completion(
        self,
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict]], 
        config: Dict
    ) -> ModelResponse:
        """Single completion for HuggingFace"""
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                **config
            )
        )
        return self._parse_response(response)
    
    def _format_tools(self, tools: List[Dict]) -> List[Dict]:
        """Format tools for HuggingFace"""
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["inputSchema"]
                }
            }
            for tool in tools
        ]
    
    def _parse_response(self, response) -> ModelResponse:
        """Parse HuggingFace response"""
        message = response.choices[0].message
        content = message.content if hasattr(message, 'content') else ""
        
        # Extract tool calls if present
        tool_calls = None
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_calls = []
            for tool_call in message.tool_calls:
                tool_calls.append({
                    "id": getattr(tool_call, 'id', f"call_{len(tool_calls)}"),
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                })
        
        # Extract thinking content
        thinking_content = self._extract_thinking(response)
        
        # Extract usage if available
        usage = None
        if hasattr(response, 'usage'):
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        
        return ModelResponse(
            content=content,
            thinking_content=thinking_content,
            tool_calls=tool_calls,
            raw_response=response,
            usage=usage,
            metadata={
                "model": self.model,
                "provider": "huggingface",
                "base_url": self.base_url
            }
        )
    
    def _extract_thinking(self, response) -> Optional[str]:
        """Extract thinking/reasoning content from HuggingFace response"""
        message = response.choices[0].message
        
        # Check for reasoning content
        if hasattr(message, 'reasoning_content'):
            return message.reasoning_content
        
        # Check for thinking in content
        content = message.content if hasattr(message, 'content') else ""
        if " thought:" in content.lower() or " reasoning:" in content.lower():
            lines = content.split('\n')
            thinking_lines = [
                line for line in lines 
                if any(word in line.lower() for word in ['thought:', 'reasoning:', 'thinking:'])
            ]
            if thinking_lines:
                return '\n'.join(thinking_lines)
        
        return None
    
    def _get_provider_info(self) -> ProviderInfo:
        """Get HuggingFace provider information"""
        return ProviderInfo(
            name="huggingface",
            supports_streaming=True,  # Model-dependent
            supports_thinking=True,
            supports_tools=True,
            supports_multimodal=False,
            models=[self.model] if self.model else []
        )

# =============================================================================
# PROVIDER FACTORY
# =============================================================================

class ProviderFactory:
    """Factory for creating provider instances"""
    
    @staticmethod
    def create_provider(config: Dict[str, Any]) -> BaseProvider:
        """Create provider from configuration"""
        provider_type = config["type"].lower()
        api_key = config.get("api_key") or os.getenv(f"{provider_type.upper()}_API_KEY")
        
        if not api_key:
            raise ValueError(f"API key required for {provider_type}")
        
        model = config.get("model")
        base_url = config.get("base_url")
        default_config = config.get("default_config", {})
        
        if provider_type == "openai":
            return OpenAIProvider(
                api_key=api_key,
                model=model or "gpt-4",
                base_url=base_url,
                **default_config
            )
        elif provider_type == "anthropic":
            return AnthropicProvider(
                api_key=api_key,
                model=model or "claude-3-sonnet-20240229",
                **default_config
            )
        elif provider_type == "gemini":
            return GeminiProvider(
                api_key=api_key,
                model=model or "gemini-pro",
                **default_config
            )
        elif provider_type == "huggingface":
            return HuggingFaceProvider(
                api_key=api_key,
                model=model,
                base_url=base_url,
                **default_config
            )
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
    
    @staticmethod
    def from_environment(provider_type: str = None) -> BaseProvider:
        """Create provider from environment variables"""
        provider_type = (provider_type or os.getenv("MODEL_PROVIDER", "openai")).lower()
        
        api_key_env = f"{provider_type.upper()}_API_KEY"
        api_key = os.getenv(api_key_env)
        
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} not set")
        
        config = {
            "type": provider_type,
            "api_key": api_key,
            "model": os.getenv(f"{provider_type.upper()}_MODEL"),
            "base_url": os.getenv(f"{provider_type.upper()}_BASE_URL"),
            "default_config": {
                "max_tokens": int(os.getenv("MODEL_MAX_TOKENS", "4096")),
                "temperature": float(os.getenv("MODEL_TEMPERATURE", "0.7")),
                "top_p": float(os.getenv("MODEL_TOP_P", "0.9")),
            }
        }
        
        return ProviderFactory.create_provider(config)
    
    @staticmethod
    def get_supported_providers() -> List[str]:
        """Get list of supported providers"""
        return ["openai", "anthropic", "gemini", "huggingface"]