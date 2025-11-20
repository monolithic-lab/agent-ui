# client.py
import json
import sys
import os
import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod

import anthropic
from openai import OpenAI
import google.generativeai as genai
from huggingface_hub import InferenceClient

from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters, ClientSession
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    content: str
    thinking_content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    raw_response: Any = None

class MCPClient:
    """MCP Client for HuggingFace Hub operations"""
    
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.available_tools: List[Dict] = []
        self._read_stream = None
        self._write_stream = None
        self._stdio_context = None
        self._connected = False
        logger.info("MCP Client initialized")
    
    async def connect(self, server_params: StdioServerParameters):
        """Connect to MCP server via stdio"""
        try:
            if self._connected:
                logger.info("MCP client already connected")
                return True
                
            logger.info(f"Connecting to MCP server: {server_params.command}")
            
            # Create stdio client connection
            self._stdio_context = stdio_client(server_params)
            self._read_stream, self._write_stream = await self._stdio_context.__aenter__()
            
            # Initialize session
            self.session = ClientSession(self._read_stream, self._write_stream)
            await self.session.__aenter__()
            
            # Initialize the connection
            await self.session.initialize()
            
            # Discover available tools
            await self.discover_tools()
            
            self._connected = True
            logger.info(f"Connected successfully. Found {len(self.available_tools)} tools")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}", exc_info=True)
            await self.disconnect()
            return False
    
    async def discover_tools(self):
        """Discover available tools from the server"""
        try:
            if not self.session:
                raise RuntimeError("Session not initialized")
            
            # List available tools
            tools_response = await self.session.list_tools()
            self.available_tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema
                }
                for tool in tools_response.tools
            ]
            
            logger.info(f"Discovered {len(self.available_tools)} tools:")
            for tool in self.available_tools:
                logger.info(f"  - {tool['name']}: {tool['description']}")
                
        except Exception as e:
            logger.error(f"Failed to discover tools: {e}", exc_info=True)
            self.available_tools = []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return results"""
        try:
            if not self.session:
                raise RuntimeError("Session not initialized")
            
            logger.info(f"Calling tool: {tool_name} with args: {arguments}")
            
            # Call the tool
            result = await self.session.call_tool(tool_name, arguments)
            
            # Parse the response
            if result.content and len(result.content) > 0:
                content = result.content[0]
                if hasattr(content, 'text'):
                    try:
                        parsed_result = json.loads(content.text)
                        logger.info(f"Tool {tool_name} executed successfully")
                        return parsed_result
                    except json.JSONDecodeError:
                        return {"result": content.text}
                else:
                    return {"result": str(content)}
            
            return {"result": "No content returned"}
            
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}", exc_info=True)
            return {"error": str(e), "tool": tool_name}
    
    def get_tools_for_model(self) -> List[Dict]:
        """Get tools formatted for model API"""
        tools = []
        for tool in self.available_tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["inputSchema"]
                }
            })
        return tools
    
    async def disconnect(self):
        """Disconnect from MCP server"""
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
                self.session = None
            
            if self._stdio_context:
                await self._stdio_context.__aexit__(None, None, None)
                self._stdio_context = None
            
            self._connected = False
            logger.info("Disconnected from MCP server")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}", exc_info=True)
    
    @property
    def is_connected(self) -> bool:
        return self._connected and self.session is not None

class ServerConnection:
    """Server connection manager"""
    
    def __init__(self, server_script_path: str = "mcp_server.py"):
        self.server_script_path = Path(server_script_path)
        self.process: Optional[asyncio.subprocess.Process] = None
        self.server_params: Optional[StdioServerParameters] = None
        logger.info(f"Server connection handler initialized for: {server_script_path}")
    
    def get_server_params(self) -> StdioServerParameters:
        """Get server parameters for stdio connection"""
        if not self.server_script_path.exists():
            raise FileNotFoundError(f"Server script not found: {self.server_script_path}")
        
        # Create server parameters for stdio connection
        self.server_params = StdioServerParameters(
            command=sys.executable,  # Python interpreter
            args=[str(self.server_script_path.absolute())],
            env=None  # Use current environment
        )
        
        logger.info(f"Server params: {self.server_params.command} {self.server_params.args}")
        return self.server_params
    
    async def validate_server(self) -> bool:
        """Validate that server script exists and is accessible"""
        try:
            if not self.server_script_path.exists():
                raise FileNotFoundError(f"Server script not found: {self.server_script_path}")
            
            if not self.server_script_path.is_file():
                raise ValueError(f"Server path is not a file: {self.server_script_path}")
            
            logger.info("Server validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Server validation failed: {e}", exc_info=True)
            return False
    
    async def health_check(self) -> bool:
        """Check if server script exists and is accessible"""
        try:
            return self.server_script_path.exists() and self.server_script_path.is_file()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.validate_server()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.process and self.process.returncode is None:
            try:
                logger.info("Server cleanup in progress...")
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Server didn't stop gracefully, killing...")
                self.process.kill()
                await self.process.wait()
            except Exception as e:
                logger.error(f"Error during server cleanup: {e}")


class ConversationManager:
    
    def __init__(self, cache_client=None, db_client=None):
        self.cache_client = cache_client
        self.db_client = db_client
        self.conversations = {}
        logger.info("ConversationManager initialized")
    
    async def get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        """Get or create a conversation session"""
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                "messages": [],
                "created_at": asyncio.get_event_loop().time(),
                "updated_at": asyncio.get_event_loop().time()
            }
            logger.info(f"Created new session: {session_id}")
        
        return self.conversations[session_id]
    
    async def get_conversation_context(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation context for model """
        if session_id not in self.conversations:
            return []

        return self.conversations[session_id]["messages"]
    
    async def add_conversation_message(self, session_id: str, role: str, content: str):
        """Add a message to conversation"""
        if session_id not in self.conversations:
            await self.get_or_create_session(session_id)
        
        self.conversations[session_id]["messages"].append({
            "role": role,
            "content": content
        })
        self.conversations[session_id]["updated_at"] = asyncio.get_event_loop().time()
    
    async def add_assistant_response(self, session_id: str, content: str):
        """Add assistant response to conversation"""
        await self.add_conversation_message(session_id, "assistant", content)
    
    async def add_tool_call(self, session_id: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Add tool call to conversation and return tool call ID"""
        tool_call_id = f"call_{int(asyncio.get_event_loop().time() * 1000)}"
        
        tool_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False)
                }
            }]
        }
        
        if session_id not in self.conversations:
            await self.get_or_create_session(session_id)
        
        self.conversations[session_id]["messages"].append(tool_message)
        return tool_call_id
    
    async def add_tool_result(self, session_id: str, tool_call_id: str, result: Any):
        """Add tool result to conversation"""
        tool_result_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
        }
        
        if session_id in self.conversations:
            self.conversations[session_id]["messages"].append(tool_result_message)

class BaseProvider(ABC):
    """Base class for all model providers with thinking support"""
    
    @abstractmethod
    async def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> ModelResponse:
        pass
    
    @abstractmethod
    def get_tools_format(self, tools: List[Dict]) -> Any:
        pass
    
    def _extract_thinking(self, response: Any) -> Optional[str]:
        return None

class OpenAIProvider(BaseProvider):
    def __init__(
        self, 
        api_key: str, 
        base_url: str = None, 
        model: str = None,
        extra_headers: Dict = None,
        default_config: Dict[str, Any] = None
    ):
        from openai import OpenAI
        
        self.client = OpenAI(
            api_key=api_key, 
            base_url=base_url,
            default_headers=extra_headers or {}
        )
        self.model = model
        self.default_config = default_config or {}
        logger.info(f"OpenAI provider initialized: {model} at {base_url or 'default'} with config {self.default_config}")
    
    async def create_chat_completion(self, messages, tools=None, **kwargs):
        # Merge default config with kwargs, with kwargs taking precedence
        config = {**self.default_config, **kwargs}
        extra_body = config.pop('extra_body', {})
        provider_tools = self.get_tools_format(tools) if tools else None
        
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=provider_tools,
                stream=False,
                extra_body=extra_body,
                **config
            )
        )
        
        return self._parse_response(response)
    
    def _parse_response(self, response) -> ModelResponse:
        message = response.choices[0].message
        content = message.content or ""
        
        thinking_content = self._extract_thinking(response)
        
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
        
        return ModelResponse(
            content=content,
            thinking_content=thinking_content,
            tool_calls=tool_calls,
            raw_response=response
        )
    
    def _extract_thinking(self, response) -> Optional[str]:
        message = response.choices[0].message
        
        if hasattr(message, 'reasoning_details') and message.reasoning_details:
            return message.reasoning_details[0].get('text', '')
        
        if hasattr(message, 'content') and isinstance(message.content, list):
            for block in message.content:
                if hasattr(block, 'type') and block.type == 'thinking':
                    return getattr(block, 'thinking', getattr(block, 'text', ''))
        
        return None
    
    def get_tools_format(self, tools):
        return [{
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["inputSchema"]
            }
        } for tool in tools] if tools else None

class AnthropicProvider(BaseProvider):
    def __init__(
        self, 
        api_key: str, 
        model: str = None,
        default_config: Dict[str, Any] = None
    ):        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.default_config = default_config or {}
        logger.info(f"Anthropic provider initialized: {model} with config {self.default_config}")
    
    async def create_chat_completion(self, messages, tools=None, **kwargs):
        config = {**self.default_config, **kwargs}
        anthropic_messages, system_message = self._convert_messages(messages)
        provider_tools = self.get_tools_format(tools) if tools else None
        
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.client.messages.create(
                model=self.model,
                system=system_message,
                messages=anthropic_messages,
                tools=provider_tools,
                **config
            )
        )
        
        return self._parse_response(response)
    
    def _parse_response(self, response) -> ModelResponse:
        thinking_content = None
        text_content = ""
        
        for block in response.content:
            if block.type == "thinking":
                thinking_content = block.thinking
            elif block.type == "text":
                text_content += block.text
        
        return ModelResponse(
            content=text_content,
            thinking_content=thinking_content,
            raw_response=response
        )
    
    def _convert_messages(self, messages):
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
    
    def get_tools_format(self, tools):
        return [{
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["inputSchema"]
        } for tool in tools]

class GeminiProvider(BaseProvider):
    def __init__(
        self, 
        api_key: str, 
        model: str = None,
        default_config: Dict[str, Any] = None
    ):
        genai.configure(api_key=api_key)
        self.model_name = model
        self.default_config = default_config or {}
        logger.info(f"Gemini provider initialized: {model}")
    
    async def create_chat_completion(self, messages, tools=None, **kwargs):
        config = {**self.default_config, **kwargs}
        
        # Convert messages to Gemini format
        gemini_messages = self._convert_messages(messages)
        
        # Configure model with tools if available
        generation_config = genai.types.GenerationConfig(**config)
        
        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config,
            tools=self.get_tools_format(tools) if tools else None
        )
        
        # Start chat or generate content
        if len(gemini_messages) > 1:
            # For multi-turn conversations
            chat = model.start_chat(history=gemini_messages[:-1])
            response = await chat.send_message_async(gemini_messages[-1])
        else:
            # Single message
            response = await model.generate_content_async(gemini_messages[0])
        
        return self._parse_response(response)
    
    def _convert_messages(self, messages: List[Dict]) -> List[Any]:
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
                        "response": {
                            "content": msg["content"]
                        }
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
    
    def _parse_response(self, response) -> ModelResponse:
        """Parse Gemini response to our standard format"""
        content = ""
        thinking_content = None
        tool_calls = None
        
        # Extract text content
        if response.text:
            content = response.text
        
        # Extract tool/function calls
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call'):
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append({
                        "id": f"call_{len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": part.function_call.name,
                            "arguments": json.dumps(part.function_call.args)
                        }
                    })
        
        return ModelResponse(
            content=content,
            thinking_content=thinking_content,
            tool_calls=tool_calls,
            raw_response=response
        )
    
    def get_tools_format(self, tools: List[Dict]) -> List[Dict]:
        """Convert tools to Gemini's function declaration format"""
        if not tools:
            return None
            
        return [{
            "function_declarations": [{
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["inputSchema"]
            } for tool in tools]
        }]
    
    def get_provider_info(self) -> Dict[str, Any]:
        return {
            "provider": "gemini",
            "model": self.model_name,
            "supports_thinking": False,
            "supports_tools": True
        }

class HuggingFaceProvider(BaseProvider):
    def __init__(
        self, 
        api_key: str, 
        model: str = None,
        base_url: str = None,
        default_config: Dict[str, Any] = None
    ):
        self.client = InferenceClient(
            api_key=api_key,
            base_url=base_url  
        )
        self.model = model
        self.default_config = default_config or {}
        logger.info(f"HuggingFace provider initialized: {model}")
    
    async def create_chat_completion(self, messages, tools=None, **kwargs):
        """Create chat completion using HuggingFace Inference API"""
        config = {**self.default_config, **kwargs}
        
        # Convert tools to HuggingFace format if available
        formatted_tools = self.get_tools_format(tools) if tools else None
        
        try:
            # HuggingFace InferenceClient is synchronous, so we run in executor
            completion = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=formatted_tools,
                    **config
                )
            )
            
            return self._parse_response(completion)
            
        except Exception as e:
            logger.error(f"HuggingFace API call failed: {e}")
            raise
    
    def _parse_response(self, completion) -> ModelResponse:
        """Parse HuggingFace response to our standard format"""
        message = completion.choices[0].message
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

        thinking_content = self._extract_thinking(completion)
        
        return ModelResponse(
            content=content,
            thinking_content=thinking_content,
            tool_calls=tool_calls,
            raw_response=completion
        )
    
    def _extract_thinking(self, completion) -> Optional[str]:
        """Extract thinking/reasoning content from HuggingFace response"""
        message = completion.choices[0].message
        
        # Check if there's reasoning content in the response
        if hasattr(message, 'reasoning_content'):
            return message.reasoning_content
        
        # For models that include thinking in the content
        content = message.content if hasattr(message, 'content') else ""
        if " thought:" in content.lower() or " reasoning:" in content.lower():
            # Simple heuristic - you might want more sophisticated parsing
            lines = content.split('\n')
            thinking_lines = [line for line in lines if any(word in line.lower() for word in ['thought:', 'reasoning:', 'thinking:'])]
            if thinking_lines:
                return '\n'.join(thinking_lines)
        
        return None
    
    def get_tools_format(self, tools: List[Dict]) -> List[Dict]:
        """Convert tools to HuggingFace's tool format"""
        if not tools:
            return None
            
        return [{
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["inputSchema"]
            }
        } for tool in tools]
    
    def get_provider_info(self) -> Dict[str, Any]:
        return {
            "provider": "huggingface",
            "model": self.model,
            "supports_thinking": True,  
            "supports_tools": True
        }
        
class ProviderFactory:
    @staticmethod
    def create_provider(provider_config: Dict[str, Any]) -> BaseProvider:
        provider_type = provider_config["type"].lower()
        api_key = provider_config.get("api_key") or os.getenv(f"{provider_type.upper()}_API_KEY")
        
        if not api_key:
            raise ValueError(f"API key required for {provider_type}")
        
        model = provider_config.get("model")
        base_url = provider_config.get("base_url")
        extra_headers = provider_config.get("extra_headers", {})
        default_config = provider_config.get("default_config", {})
        
        if provider_type == "openai":
            return OpenAIProvider(
                api_key=api_key, 
                base_url=base_url, 
                model=model, 
                extra_headers=extra_headers,
                default_config=default_config
            )
        elif provider_type == "anthropic":
            return AnthropicProvider(
                api_key=api_key, 
                model=model,
                default_config=default_config
            )
        elif provider_type == "gemini":
            return GeminiProvider(
                api_key=api_key,
                model=model,
                default_config=default_config
            )
        elif provider_type == "huggingface":
            return HuggingFaceProvider(
                api_key=api_key,
                model=model,
                base_url=base_url,
                default_config=default_config
            )
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
    
    @staticmethod
    def from_environment() -> BaseProvider:
        provider_type = os.getenv("MODEL_PROVIDER", "openai").lower()
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


class MCPClientSystem:
    """MCP client system"""
    
    def __init__(
        self, 
        provider_config: Dict[str, Any],
        server_script_path: str = "mcp_server.py", 
        cache_client=None, 
        db_client=None
    ):
        self.provider = ProviderFactory.create_provider(provider_config)
        self.provider_config = provider_config
        self.mcp_client = MCPClient()
        self.server_connection = ServerConnection(server_script_path)
        self.conversation_manager = ConversationManager(
            cache_client=cache_client,
            db_client=db_client
        )
        
        logger.info(f"MCP Client System initialized with {provider_config['type']}")
    
    async def connect(self) -> bool:
        """Connect all system components"""
        try:
            # Validate server connection
            if not await self.server_connection.validate_server():
                return False
            
            # Connect to MCP server
            server_params = self.server_connection.get_server_params()
            if not await self.mcp_client.connect(server_params):
                return False
            
            logger.info("MCP Client System connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect MCP Client System: {e}", exc_info=True)
            await self.disconnect()
            return False
    
    async def disconnect(self):
        """Disconnect all system components"""
        try:
            # Disconnect MCP client
            await self.mcp_client.disconnect()
            logger.info("MCP Client System disconnected")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}", exc_info=True)
    
    async def _call_model(self, messages: List[Dict], tools: Optional[List] = None) -> ModelResponse:
        """Call the model and return structured response with thinking"""
        try:
            return await self.provider.create_chat_completion(
                messages=messages,
                tools=tools
            )
        except Exception as e:
            logger.error(f"Model call failed: {e}")
            raise
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()

    async def process_message(
        self,
        session_id: str,
        message: str,
        use_streaming: bool = False
    ) -> Dict[str, Any]:
        """Process a user message with thinking support"""
        try:
            await self.conversation_manager.get_or_create_session(session_id)

            # Add user message
            await self.conversation_manager.add_conversation_message(
                session_id, "user", message
            )

            # Prepare tools
            tools = self.mcp_client.get_tools_for_model()

            # Get conversation context
            context = await self.conversation_manager.get_conversation_context(session_id)

            # FIRST MODEL CALL
            logger.info(f"First model call with {len(context)} messages")
            response = await self._call_model(context, tools if tools else None)

            # Store thinking content if available
            thinking_content = response.thinking_content
            
            # Add assistant response to conversation
            assistant_message = {"role": "assistant", "content": response.content or ""}
            if response.tool_calls:
                assistant_message["tool_calls"] = response.tool_calls
            
            if session_id in self.conversation_manager.conversations:
                self.conversation_manager.conversations[session_id]["messages"].append(assistant_message)

            # Execute tool calls if any
            final_content = response.content
            tool_results = []
            
            if response.tool_calls:
                logger.info(f"Executing {len(response.tool_calls)} tool calls")
                
                for tool_call in response.tool_calls:
                    tool_name = tool_call["function"]["name"]
                    arguments = json.loads(tool_call["function"]["arguments"])

                    # Execute tool
                    result = await self.mcp_client.call_tool(tool_name, arguments)

                    # Add tool result to conversation
                    tool_result_message = {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
                    }
                    
                    if session_id in self.conversation_manager.conversations:
                        self.conversation_manager.conversations[session_id]["messages"].append(tool_result_message)

                    tool_results.append({
                        "tool_name": tool_name,
                        "result": result
                    })

                # SECOND MODEL CALL after tool execution
                updated_context = await self.conversation_manager.get_conversation_context(session_id)
                logger.info(f"Second model call with {len(updated_context)} messages")

                final_response = await self._call_model(updated_context, tools=tools)
                final_content = final_response.content

                # Add final response to conversation
                if final_content:
                    await self.conversation_manager.add_assistant_response(session_id, final_content)

            return {
                "session_id": session_id,
                "content": final_content,
                "thinking_content": thinking_content,
                "tool_calls_executed": len(response.tool_calls) if response.tool_calls else 0,
                "tool_results": tool_results,
                "has_final_response": bool(final_content and final_content.strip()),
                "provider_type": self.provider_config["type"]
            }

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return {
                "session_id": session_id,
                "error": str(e)
            }