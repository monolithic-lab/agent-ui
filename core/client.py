# core/client.py
import os
import sys
import uuid
import json
import asyncio
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING

import anthropic
from openai import OpenAI, AsyncOpenAI
import google.generativeai as genai
from huggingface_hub import InferenceClient, AsyncInferenceClient
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters, ClientSession
from typing import Dict, Any, Optional, List, AsyncGenerator, Union

from core.database import Session, Message, UserSettings
from core.provider import (
    ModelResponse,
    ProviderInfo,
    BaseProvider,
    OpenAIProvider, 
    AnthropicProvider, 
    GeminiProvider, 
    HuggingFaceProvider,
    ProviderFactory
)

logger = logging.getLogger(__name__)


class MessageType(str):
    USER = "user"
    ASSISTANT = "assistant" 
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SYSTEM = "system"


class WebSocketManager:
    """WebSocket manager for real-time updates (placeholder implementation)"""
    
    def __init__(self):
        self.connections: Dict[str, set] = {}  # session_id -> set of connections
        
    async def connect(self, connection_id: str, session_id: str):
        """Handle new connection (placeholder)"""
        if session_id not in self.connections:
            self.connections[session_id] = set()
        self.connections[session_id].add(connection_id)
        logger.info(f"WebSocket connected: {session_id}")
    
    def disconnect(self, connection_id: str, session_id: str):
        """Handle disconnection (placeholder)"""
        if session_id in self.connections:
            self.connections[session_id].discard(connection_id)
            if not self.connections[session_id]:
                del self.connections[session_id]
        logger.info(f"WebSocket disconnected: {session_id}")
    
    async def send_update(self, session_id: str, message: Dict[str, Any]):
        """Send update to all connections for a session (placeholder)"""
        if session_id in self.connections:
            # In real implementation, send to actual WebSocket connections
            logger.info(f"Broadcasting to {len(self.connections[session_id])} connections: {message}")
        
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
    """conversation manager with database persistence and tool support"""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.memory_cache = {}
        
    async def create_session(
        self, 
        user_id: str, 
        title: str = None, 
        model: str = "gpt-4", 
        provider: str = "openai",
        settings: Dict = None
    ) -> str:
        """Create new conversation session"""
        session_id = str(uuid.uuid4())
        session = Session(
            id=session_id,
            user_id=user_id,
            title=title or "New Conversation",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            model=model,
            provider=provider,
            settings=settings or {}
        )
        
        await self.db.save_session(session)
        self.memory_cache[session_id] = session
        logger.info(f"Created session {session_id} for user {user_id}")
        return session_id
    
    async def get_or_create_session(self, session_id: str, user_id: str) -> Session:
        """Get existing session or create new one"""
        # Try cache first
        if session_id in self.memory_cache:
            return self.memory_cache[session_id]
        
        # Try database
        session = await self.db.get_session(session_id)
        if session and session.user_id == user_id:
            self.memory_cache[session_id] = session
            return session
        
        # Create new session
        new_session_id = await self.create_session(user_id)
        return await self.get_or_create_session(new_session_id, user_id)
    
    async def save_message(self, session_id: str, message: Message):
        """Save message to database and cache"""
        await self.db.save_message(message)
        
        # Update memory cache and session metadata
        if session_id in self.memory_cache:
            session = self.memory_cache[session_id]
            session.message_count += 1
            session.updated_at = datetime.now(timezone.utc)
            
            # Cache messages for fast access
            if "_messages" not in session:
                session._messages = []
            session._messages.append(message)
    
    async def get_conversation_context(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation context for model (with tool support)"""
        messages = await self.get_conversation_messages(session_id, limit)
        
        # Convert to OpenAI-compatible format with tool support
        context = []
        for msg in messages:
            if hasattr(msg, 'to_openai_format'):
                # If Message model has conversion method
                context.append(msg.to_openai_format())
            else:
                # Basic conversion
                message_data = {
                    "role": msg.role,
                    "content": msg.content
                }
                
                # Add tool calls if present (from first class)
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    message_data["tool_calls"] = msg.tool_calls
                
                # Add tool call ID if present (from first class)
                if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                    message_data["tool_call_id"] = msg.tool_call_id
                
                context.append(message_data)
        
        return context
    
    async def get_conversation_messages(self, session_id: str, limit: int = 50) -> List[Message]:
        """Get conversation messages for LLM context"""
        # Check cache first
        if session_id in self.memory_cache and hasattr(self.memory_cache[session_id], '_messages'):
            cached_messages = self.memory_cache[session_id]._messages
            if len(cached_messages) >= limit:
                return cached_messages[-limit:]
        
        # Get from database
        messages = await self.db.get_messages(session_id, limit)
        
        # Update cache
        if session_id in self.memory_cache:
            self.memory_cache[session_id]._messages = messages
        
        return messages
    
    async def add_conversation_message(self, session_id: str, role: str, content: str, **kwargs) -> Message:
        """Add a message to conversation"""
        message = Message(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role=role,
            content=content,
            timestamp=datetime.now(timezone.utc),
            **kwargs
        )
        
        await self.save_message(session_id, message)
        logger.info(f"Added {role} message to session {session_id}")
        return message
    
    async def add_assistant_response(self, session_id: str, content: str, **kwargs) -> Message:
        """Add assistant response to conversation (from first class)"""
        return await self.add_conversation_message(session_id, "assistant", content, **kwargs)
    
    async def add_tool_call(self, session_id: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Add tool call to conversation and return tool call ID (from first class)"""
        tool_call_id = f"call_{int(datetime.now(timezone.utc).timestamp() * 1000)}"
        
        tool_message = Message(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="assistant",
            content="",
            tool_calls=[{
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False)
                }
            }],
            timestamp=datetime.now(timezone.utc)
        )
        
        await self.save_message(session_id, tool_message)
        logger.info(f"Added tool call {tool_name} to session {session_id}")
        return tool_call_id
    
    async def add_tool_result(self, session_id: str, tool_call_id: str, result: Any, **kwargs) -> Message:
        """Add tool result to conversation (from first class)"""
        content = json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
        
        tool_result_message = Message(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            timestamp=datetime.now(timezone.utc),
            **kwargs
        )
        
        await self.save_message(session_id, tool_result_message)
        logger.info(f"Added tool result for call {tool_call_id} to session {session_id}")
        return tool_result_message
    
    async def get_session_messages_with_tools(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get messages including tool calls and results in OpenAI format"""
        return await self.get_conversation_context(session_id, limit)
    
    async def cleanup_old_messages(self, session_id: str, keep_last: int = 100):
        """Clean up old messages while keeping recent ones"""
        await self.db.cleanup_old_messages(session_id, keep_last)
        
        # Clear cache to force refresh
        if session_id in self.memory_cache and hasattr(self.memory_cache[session_id], '_messages'):
            del self.memory_cache[session_id]._messages
    
    async def delete_session(self, session_id: str):
        """Delete session and all related data"""
        await self.db.delete_session(session_id)
        if session_id in self.memory_cache:
            del self.memory_cache[session_id]
    
    async def get_user_sessions(self, user_id: str, limit: int = 20) -> List[Session]:
        """Get all sessions for a user"""
        return await self.db.get_user_sessions(user_id, limit)
    
    async def update_session_title(self, session_id: str, title: str):
        """Update session title"""
        await self.db.update_session_title(session_id, title)
        if session_id in self.memory_cache:
            self.memory_cache[session_id].title = title
            self.memory_cache[session_id].updated_at = datetime.now(timezone.utc)

class MCPClient:
    """
    MCP Client with multi-server support, health checks, and proper resource management.
    Combines the best features from both single-server and multi-server implementations.
    """
    
    def __init__(self):
        self.sessions: Dict[str, Any] = {}  # Multiple MCP server sessions
        self.available_tools: Dict[str, List[Dict]] = {}  # Tools per server
        self.server_configs: Dict[str, Dict] = {}  # Server configurations
        self.health_status: Dict[str, bool] = {}
        self._stdio_contexts: Dict[str, Any] = {}  # Store stdio contexts for proper cleanup
        self._connected_servers = set()
        
        logger.info("MCP Client initialized")

    async def add_mcp_server(self, name: str, config: Dict[str, Any]):
        """Add MCP server configuration"""
        self.server_configs[name] = config
        logger.info(f"Added MCP server configuration: {name}")

    async def connect_to_server(self, server_name: str) -> bool:
        """Connect to a specific MCP server with proper resource management"""
        if server_name in self._connected_servers:
            logger.info(f"Already connected to {server_name}")
            return True
            
        config = self.server_configs.get(server_name)
        if not config:
            logger.error(f"Server {server_name} not configured")
            return False
        
        try:
            server_params = StdioServerParameters(
                command=config.get("command", "python"),
                args=config.get("args", []),
                env=config.get("env", os.environ.copy())
            )
            
            # Create and store stdio context for proper cleanup
            stdio_context = stdio_client(server_params)
            read_stream, write_stream = await stdio_context.__aenter__()
            self._stdio_contexts[server_name] = stdio_context
            
            # Create session
            session = ClientSession(read_stream, write_stream)
            await session.__aenter__()
            await session.initialize()
            
            # Store session and discover tools
            self.sessions[server_name] = session
            await self._discover_tools(server_name)
            
            self._connected_servers.add(server_name)
            self.health_status[server_name] = True
            
            logger.info(f"Connected to MCP server: {server_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to {server_name}: {e}")
            await self._cleanup_failed_connection(server_name)
            self.health_status[server_name] = False
            return False

    async def _discover_tools(self, server_name: str):
        """Discover tools from MCP server"""
        try:
            session = self.sessions[server_name]
            tools_response = await session.list_tools()
            
            self.available_tools[server_name] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema,
                    "server": server_name
                }
                for tool in tools_response.tools
            ]
            
            logger.info(f"Discovered {len(self.available_tools[server_name])} tools from {server_name}")
            
        except Exception as e:
            logger.error(f"Failed to discover tools from {server_name}: {e}")
            self.available_tools[server_name] = []

    async def _cleanup_failed_connection(self, server_name: str):
        """Clean up resources from failed connection"""
        if server_name in self._stdio_contexts:
            try:
                await self._stdio_contexts[server_name].__aexit__(None, None, None)
                del self._stdio_contexts[server_name]
            except Exception as e:
                logger.error(f"Error cleaning up {server_name}: {e}")

    async def health_check(self, server_name: str) -> Dict[str, Any]:
        """Perform health check on MCP server"""
        if server_name not in self.sessions:
            return {"status": "disconnected", "server": server_name}
        
        try:
            # Simple ping by listing tools
            session = self.sessions[server_name]
            await session.list_tools()
            self.health_status[server_name] = True
            return {
                "status": "healthy",
                "server": server_name,
                "tools_count": len(self.available_tools.get(server_name, []))
            }
        except Exception as e:
            self.health_status[server_name] = False
            return {
                "status": "unhealthy", 
                "server": server_name,
                "error": str(e)
            }

    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all connected servers"""
        results = {}
        for server_name in list(self.sessions.keys()):
            results[server_name] = await self.health_check(server_name)
        return results

    def get_all_tools(self) -> List[Dict]:
        """Get all available tools from all connected servers"""
        all_tools = []
        for server_name, tools in self.available_tools.items():
            for tool in tools:
                tool_copy = tool.copy()
                tool_copy["server"] = server_name
                all_tools.append(tool_copy)
        return all_tools

    def get_tools_for_server(self, server_name: str) -> List[Dict]:
        """Get tools for a specific server"""
        return self.available_tools.get(server_name, [])

    def get_tools_for_model(self, server_name: Optional[str] = None) -> List[Dict]:
        """Get tools formatted for model API"""
        tools = []
        
        if server_name:
            # Get tools from specific server
            server_tools = self.available_tools.get(server_name, [])
            for tool in server_tools:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["inputSchema"]
                    }
                })
        else:
            # Get all tools from all servers
            for server_tools in self.available_tools.values():
                for tool in server_tools:
                    tools.append({
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool["description"],
                            "parameters": tool["inputSchema"]
                        }
                    })
        
        return tools

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call tool on specific server"""
        if server_name not in self.sessions:
            raise RuntimeError(f"Not connected to server {server_name}")
        
        session = self.sessions[server_name]
        try:
            logger.info(f"Calling tool {tool_name} on server {server_name} with args: {arguments}")
            
            result = await session.call_tool(tool_name, arguments)
            
            if result.content and len(result.content) > 0:
                content = result.content[0]
                if hasattr(content, 'text'):
                    try:
                        parsed_result = json.loads(content.text)
                        logger.info(f"Tool {tool_name} on {server_name} executed successfully")
                        return parsed_result
                    except json.JSONDecodeError:
                        return {"result": content.text}
                else:
                    return {"result": str(content)}
            return {"result": "No content returned"}
            
        except Exception as e:
            logger.error(f"Tool call failed on {server_name}: {e}")
            return {"error": str(e), "tool": tool_name, "server": server_name}

    async def call_tool_with_fallback(self, tool_name: str, arguments: Dict[str, Any], preferred_server: Optional[str] = None) -> Dict[str, Any]:
        """Call tool with server fallback logic"""
        servers_to_try = []
        
        if preferred_server and preferred_server in self.sessions:
            servers_to_try.append(preferred_server)
        
        # Add other servers that have this tool
        for server_name, tools in self.available_tools.items():
            if server_name != preferred_server and any(tool["name"] == tool_name for tool in tools):
                servers_to_try.append(server_name)
        
        # Try servers in order
        for server_name in servers_to_try:
            try:
                result = await self.call_tool(server_name, tool_name, arguments)
                if "error" not in result:
                    return result
            except Exception as e:
                logger.warning(f"Tool call failed on {server_name}, trying next server: {e}")
                continue
        
        return {"error": f"Tool {tool_name} not available on any connected server"}

    async def disconnect_server(self, server_name: str):
        """Disconnect from specific server"""
        try:
            if server_name in self.sessions:
                await self.sessions[server_name].__aexit__(None, None, None)
                del self.sessions[server_name]
            
            if server_name in self._stdio_contexts:
                await self._stdio_contexts[server_name].__aexit__(None, None, None)
                del self._stdio_contexts[server_name]
                
            if server_name in self.health_status:
                del self.health_status[server_name]
            
            if server_name in self._connected_servers:
                self._connected_servers.remove(server_name)
                
            logger.info(f"Disconnected from MCP server: {server_name}")
            
        except Exception as e:
            logger.error(f"Error disconnecting from {server_name}: {e}")

    async def disconnect_all(self):
        """Disconnect from all MCP servers with proper resource cleanup"""
        for server_name in list(self.sessions.keys()):
            await self.disconnect_server(server_name)
        
        logger.info("Disconnected from all MCP servers")

    def is_connected(self, server_name: str) -> bool:
        """Check if connected to specific server"""
        return server_name in self.sessions and server_name in self._connected_servers

    def get_connected_servers(self) -> List[str]:
        """Get list of all connected server names"""
        return list(self._connected_servers)

    def get_server_tool_names(self, server_name: str) -> List[str]:
        """Get list of tool names for a specific server"""
        tools = self.available_tools.get(server_name, [])
        return [tool["name"] for tool in tools]

    def has_tool(self, tool_name: str, server_name: Optional[str] = None) -> bool:
        """Check if tool is available"""
        if server_name:
            return any(tool["name"] == tool_name for tool in self.available_tools.get(server_name, []))
        else:
            return any(
                any(tool["name"] == tool_name for tool in tools)
                for tools in self.available_tools.values()
            )

    async def reconnect_server(self, server_name: str) -> bool:
        """Reconnect to a server"""
        await self.disconnect_server(server_name)
        return await self.connect_to_server(server_name)

    @property
    def total_tools_count(self) -> int:
        """Get total number of available tools across all servers"""
        return sum(len(tools) for tools in self.available_tools.values())

    @property
    def connected_servers_count(self) -> int:
        """Get number of connected servers"""
        return len(self._connected_servers)

class MCPClientSystem:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db = DatabaseManager(
            config.get("mongodb_url"),
            config.get("db_name", "agent_ui")
        )
        self.mcp_client = MCPClient()
        self.conversation_manager = None 
        self.websocket_manager = WebSocketManager()
        self.providers = {}
        
    async def initialize(self):
        """Initialize the client system"""
        # Connect to database
        await self.db.connect()
        self.conversation_manager = ConversationManager(self.db)
        
        # Initialize MCP servers
        await self._setup_mcp_servers()
        
        # Initialize providers
        await self._setup_providers()
        
        logger.info("Agent Client initialized")
    
    async def _setup_mcp_servers(self):
        """Setup MCP servers from configuration"""
        mcp_servers = self.config.get("mcp_servers", [])
        
        for server_config in mcp_servers:
            await self.mcp_client.add_mcp_server(
                server_config["name"], 
                server_config
            )
            
            # Auto-connect to servers if configured
            if server_config.get("auto_connect", True):
                await self.mcp_client.connect_to_server(server_config["name"])
    
    async def _setup_providers(self):
        """Initialize LLM providers"""
        provider_configs = self.config.get("providers", {})
        
        for name, config in provider_configs.items():
            try:
                self.providers[name] = ProviderFactory.create_provider(config)
                logger.info(f"Initialized provider: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize provider {name}: {e}")
    
    async def create_session(self, user_id: str, **kwargs) -> str:
        """Create new conversation session"""
        return await self.conversation_manager.create_session(user_id, **kwargs)
    
    async def process_message(
        self,
        session_id: str,
        user_id: str,
        message: str,
        use_streaming: bool = False,
    ) -> Union[ModelResponse, AsyncGenerator[ModelResponse, None]]:
        
        # Ensure session exists
        session = await self.conversation_manager.get_or_create_session(session_id, user_id)
        
        # Save user message
        user_message = Message(
            id=str(uuid.uuid4()),
            role="user",
            content=message,
            timestamp=datetime.now(timezone.utc),
            session_id=session_id
        )
        await self.conversation_manager.save_message(session_id, user_message)
        
        # Get conversation context
        messages = await self.conversation_manager.get_conversation_messages(session_id)
        
        # Check for available tools
        tools = self.mcp_client.get_all_tools()
        has_tools = len(tools) > 0
        
        # Get provider
        provider = self.providers.get(session.provider)
        if not provider:
            raise ValueError(f"Provider {session.provider} not configured")
        
        # Send WebSocket update for message processing start
        await self.websocket_manager.send_update(session_id, {
            "type": "message_processing",
            "status": "started",
        })
        
        if use_streaming and hasattr(provider, 'create_chat_completion'):
            # Streaming response
            async for response in provider.create_chat_completion(
                messages=[{"role": msg.role, "content": msg.content} for msg in messages],
                tools=tools,
                stream=True
            ):
                # Send streaming update
                await self.websocket_manager.send_update(session_id, {
                    "type": "stream_chunk",
                    "content": response.content
                })
                yield response
        else:
            # Single response processing
            response = await provider.create_chat_completion(
                messages=[{"role": msg.role, "content": msg.content} for msg in messages],
                tools=tools
            )
            
            # Handle tool calls
            if response.tool_calls:
                await self._handle_tool_calls(session_id, response.tool_calls, tools)
                
                # Get updated context after tool execution
                updated_messages = await self.conversation_manager.get_conversation_messages(session_id)
                final_response = await provider.create_chat_completion(
                    messages=[{"role": msg.role, "content": msg.content} for msg in updated_messages],
                    tools=tools
                )
                response = final_response
            
            # Save assistant response
            assistant_message = Message(
                id=str(uuid.uuid4()),
                role="assistant", 
                content=response.content,
                timestamp=datetime.now(timezone.utc),
                session_id=session_id,
                tool_calls=response.tool_calls,
                thinking_content=response.thinking_content
            )
            await self.conversation_manager.save_message(session_id, assistant_message)
            
            # Send completion update
            await self.websocket_manager.send_update(session_id, {
                "type": "message_complete",
                "response": asdict(response)
            })
            
            yield response
    
    async def _handle_tool_calls(self, session_id: str, tool_calls: List[Dict], tools: List[Dict]):
        """Handle tool calls with progress updates"""
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            
            # Find which server has this tool
            tool_info = next((t for t in tools if t["name"] == tool_name), None)
            if not tool_info:
                continue
            
            server_name = tool_info["server"]
            
            # Send tool call start
            await self.websocket_manager.send_update(session_id, {
                "type": "tool_call_start",
                "tool_name": tool_name,
                "arguments": arguments
            })
            
            try:
                # Execute tool
                result = await self.mcp_client.call_tool(server_name, tool_name, arguments)
                
                # Send tool result
                await self.websocket_manager.send_update(session_id, {
                    "type": "tool_result",
                    "tool_name": tool_name,
                    "result": result
                })
                
                # Save tool result message
                tool_result_message = Message(
                    id=str(uuid.uuid4()),
                    role="tool",
                    content=json.dumps(result) if isinstance(result, (dict, list)) else str(result),
                    timestamp=datetime.now(timezone.utc),
                    session_id=session_id,
                    metadata={"tool_call_id": tool_call["id"], "server": server_name}
                )
                await self.conversation_manager.save_message(session_id, tool_result_message)
                
            except Exception as e:
                # Send error
                await self.websocket_manager.send_update(session_id, {
                    "type": "tool_error",
                    "tool_name": tool_name,
                    "error": str(e)
                })
    
    async def get_session_history(self, session_id: str, user_id: str) -> Dict[str, Any]:
        """Get session history and metadata"""
        session = await self.conversation_manager.get_or_create_session(session_id, user_id)
        messages = await self.conversation_manager.get_conversation_messages(session_id)
        
        return {
            "session": asdict(session),
            "messages": [asdict(msg) for msg in messages],
            "total_messages": len(messages)
        }
    
    async def get_user_sessions(self, user_id: str) -> List[Session]:
        """Get all sessions for a user"""
        return await self.db.get_user_sessions(user_id)
    
    async def get_user_settings(self, user_id: str) -> UserSettings:
        """Get user settings"""
        settings = await self.db.get_user_settings(user_id)
        if not settings:
            settings = UserSettings(user_id=user_id)
            await self.db.save_user_settings(settings)
        return settings
    
    async def save_user_settings(self, settings: UserSettings):
        """Save user settings"""
        await self.db.save_user_settings(settings)
    
    async def health_check(self) -> Dict[str, Any]:
        """System health check"""
        health = {
            "database": False,
            "mcp_servers": {},
            "providers": list(self.providers.keys()),
            "mcp_connected": len(self.mcp_client._connected_servers)
        }
        
        # Check database
        try:
            await self.db.db.admin.command('ping')
            health["database"] = True
        except:
            pass
        
        # Check MCP servers
        for server_name in self.mcp_client.server_configs:
            health["mcp_servers"][server_name] = await self.mcp_client.health_check(server_name)
        
        return health
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.mcp_client.disconnect_all()
        await self.db.disconnect()