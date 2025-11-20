# tools/mcp_manager.py
"""
MCP Manager Singleton
Centralized management of MCP (Model Context Protocol) servers
"""

import asyncio
import logging
import json
import os
import signal
import subprocess
import time
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)

class MCPConnectionStatus(Enum):
    """MCP connection status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class MCPServerConfig:
    """Configuration for MCP server"""
    name: str
    command: List[str]
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 5.0
    auto_restart: bool = True
    health_check_interval: float = 30.0
    connection_timeout: float = 10.0


@dataclass
class MCPConnection:
    """MCP server connection information"""
    server_id: str
    config: MCPServerConfig
    status: MCPConnectionStatus = MCPConnectionStatus.DISCONNECTED
    process: Optional[subprocess.Popen] = None
    last_health_check: float = field(default_factory=time.time)
    connection_attempts: int = 0
    error_messages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_healthy(self) -> bool:
        """Check if connection is healthy"""
        return (
            self.status == MCPConnectionStatus.CONNECTED and
            time.time() - self.last_health_check < self.config.health_check_interval
        )
    
    def get_uptime(self) -> float:
        """Get connection uptime in seconds"""
        if self.process and self.status == MCPConnectionStatus.CONNECTED:
            return time.time() - self.last_health_check
        return 0.0


class MCPManager:
    """
    Enterprise-grade MCP Manager Singleton
    Provides centralized management of all MCP servers
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize MCP Manager (only once due to singleton)"""
        if self._initialized:
            return
            
        self._initialized = True
        self._connections: Dict[str, MCPConnection] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Statistics
        self._stats = {
            'total_connections': 0,
            'active_connections': 0,
            'failed_connections': 0,
            'total_restarts': 0,
            'avg_connection_time': 0.0,
            'total_uptime': 0.0
        }
        
        logger.info("MCPManager singleton initialized")
    
    async def start(self):
        """Start the MCP Manager"""
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info("MCP Manager started")
    
    async def stop(self):
        """Stop the MCP Manager"""
        self._shutdown_event.set()
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Terminate all connections
        for server_id in list(self._connections.keys()):
            await self.disconnect(server_id)
        
        logger.info("MCP Manager stopped")
    
    async def register_server(self, config: MCPServerConfig) -> str:
        """
        Register an MCP server configuration
        
        Args:
            config: MCP server configuration
        
        Returns:
            server_id: Unique server identifier
        """
        server_id = str(uuid.uuid4())
        
        connection = MCPConnection(
            server_id=server_id,
            config=config,
            metadata={
                'registered_at': time.time(),
                'config_hash': hash(str(config))
            }
        )
        
        self._connections[server_id] = connection
        self._stats['total_connections'] += 1
        
        logger.info("Registered MCP server: %s (%s)", config.name, server_id)
        await self._emit_event('server_registered', {'server_id': server_id, 'config': config})
        
        return server_id
    
    async def connect(self, server_id: str) -> bool:
        """
        Connect to an MCP server
        
        Args:
            server_id: Server identifier
        
        Returns:
            bool: True if connection successful
        """
        if server_id not in self._connections:
            logger.error("Server %s not found", server_id)
            return False
        
        connection = self._connections[server_id]
        
        if connection.status == MCPConnectionStatus.CONNECTED:
            logger.info("Server %s already connected", server_id)
            return True
        
        try:
            connection.status = MCPConnectionStatus.CONNECTING
            connection.connection_attempts += 1
            
            logger.info("Connecting to MCP server %s", connection.config.name)
            
            # Start the server process
            process = await asyncio.create_subprocess_exec(
                *connection.config.command,
                *connection.config.args,
                cwd=connection.config.cwd,
                env={**os.environ, **connection.config.env},
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            connection.process = process
            connection.status = MCPConnectionStatus.CONNECTED
            connection.last_health_check = time.time()
            connection.error_messages.clear()
            
            self._stats['active_connections'] += 1
            
            # Start monitoring task for this connection
            asyncio.create_task(self._monitor_connection(server_id))
            
            logger.info("Successfully connected to MCP server %s", connection.config.name)
            await self._emit_event('server_connected', {
                'server_id': server_id,
                'config': connection.config
            })
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to connect to server {server_id}: {str(e)}"
            connection.status = MCPConnectionStatus.ERROR
            connection.error_messages.append(error_msg)
            self._stats['failed_connections'] += 1
            
            logger.error(error_msg)
            await self._emit_event('server_error', {
                'server_id': server_id,
                'error': str(e)
            })
            
            return False
    
    async def disconnect(self, server_id: str) -> bool:
        """
        Disconnect from an MCP server
        
        Args:
            server_id: Server identifier
        
        Returns:
            bool: True if disconnection successful
        """
        if server_id not in self._connections:
            logger.error("Server %s not found", server_id)
            return False
        
        connection = self._connections[server_id]
        
        try:
            connection.status = MCPConnectionStatus.TERMINATED
            
            # Terminate the process
            if connection.process:
                try:
                    connection.process.terminate()
                    await asyncio.wait_for(connection.process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    connection.process.kill()
                    await connection.process.wait()
                except Exception as e:
                    logger.error("Error terminating process for server %s: %s", server_id, e)
                
                connection.process = None
            
            if connection.status == MCPConnectionStatus.CONNECTED:
                self._stats['active_connections'] -= 1
            
            logger.info("Disconnected from MCP server %s", connection.config.name)
            await self._emit_event('server_disconnected', {
                'server_id': server_id,
                'config': connection.config
            })
            
            return True
            
        except Exception as e:
            logger.error("Error disconnecting from server %s: %s", server_id, e)
            return False
    
    async def send_message(self, server_id: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Send message to MCP server
        
        Args:
            server_id: Server identifier
            message: Message to send
        
        Returns:
            Optional[Dict[str, Any]]: Response from server
        """
        if server_id not in self._connections:
            logger.error("Server %s not found", server_id)
            return None
        
        connection = self._connections[server_id]
        
        if connection.status != MCPConnectionStatus.CONNECTED:
            logger.error("Server %s is not connected", server_id)
            return None
        
        try:
            # Convert message to JSON
            message_data = json.dumps(message) + '\n'
            message_bytes = message_data.encode()
            
            # Send message
            if connection.process and connection.process.stdin:
                connection.process.stdin.write(message_bytes)
                await connection.process.stdin.drain()
            
            # Wait for response (simplified - would need proper MCP protocol implementation)
            if connection.process and connection.process.stdout:
                response_line = await asyncio.wait_for(
                    connection.process.stdout.readline(),
                    timeout=connection.config.connection_timeout
                )
                
                if response_line:
                    response = json.loads(response_line.decode().strip())
                    return response
            
            return None
            
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for response from server %s", server_id)
            return None
        except Exception as e:
            logger.error("Error sending message to server %s: %s", server_id, e)
            return None
    
    async def restart_server(self, server_id: str) -> bool:
        """
        Restart an MCP server
        
        Args:
            server_id: Server identifier
        
        Returns:
            bool: True if restart successful
        """
        logger.info("Restarting MCP server %s", server_id)
        
        # Disconnect first
        await self.disconnect(server_id)
        
        # Wait a bit
        await asyncio.sleep(1.0)
        
        # Reconnect
        success = await self.connect(server_id)
        
        if success:
            self._stats['total_restarts'] += 1
            logger.info("Successfully restarted MCP server %s", server_id)
        
        return success
    
    def get_server_status(self, server_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of an MCP server
        
        Args:
            server_id: Server identifier
        
        Returns:
            Optional[Dict[str, Any]]: Server status information
        """
        if server_id not in self._connections:
            return None
        
        connection = self._connections[server_id]
        
        return {
            'server_id': server_id,
            'name': connection.config.name,
            'status': connection.status.value,
            'is_healthy': connection.is_healthy(),
            'uptime': connection.get_uptime(),
            'connection_attempts': connection.connection_attempts,
            'error_messages': connection.error_messages,
            'last_health_check': connection.last_health_check,
            'metadata': connection.metadata
        }
    
    def list_servers(self) -> List[Dict[str, Any]]:
        """List all registered servers"""
        return [
            self.get_server_status(server_id)
            for server_id in self._connections.keys()
        ]
    
    async def on_event(self, event_type: str, handler: Callable):
        """
        Register event handler
        
        Args:
            event_type: Type of event
            handler: Event handler function
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        
        self._event_handlers[event_type].append(handler)
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to all registered handlers"""
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error("Error in event handler for %s: %s", event_type, e)
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_health_checks()
                await asyncio.sleep(10.0)  # Check every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in health check loop: %s", e)
    
    async def _perform_health_checks(self):
        """Perform health checks on all connections"""
        current_time = time.time()
        
        for server_id, connection in list(self._connections.items()):
            if connection.status == MCPConnectionStatus.CONNECTED:
                # Check if process is still running
                if connection.process and connection.process.returncode is not None:
                    # Process has terminated unexpectedly
                    logger.warning("MCP server %s process terminated unexpectedly", server_id)
                    connection.status = MCPConnectionStatus.ERROR
                    connection.error_messages.append("Process terminated unexpectedly")
                    
                    # Attempt auto-restart if enabled
                    if connection.config.auto_restart:
                        await asyncio.sleep(connection.config.retry_delay)
                        await self.restart_server(server_id)
                
                # Check health check interval
                if current_time - connection.last_health_check > connection.config.health_check_interval:
                    connection.last_health_check = current_time
                    
                    # Send health check message (simplified)
                    try:
                        health_response = await self.send_message(server_id, {
                            'type': 'health_check',
                            'timestamp': current_time
                        })
                        
                        if health_response is None:
                            logger.warning("Health check failed for server %s", server_id)
                            
                    except Exception as e:
                        logger.error("Health check error for server %s: %s", server_id, e)
    
    async def _monitor_connection(self, server_id: str):
        """Monitor individual connection"""
        connection = self._connections[server_id]
        
        try:
            while (connection.status == MCPConnectionStatus.CONNECTED and
                   connection.process and connection.process.returncode is None):
                
                # Read stderr for error messages
                if connection.process.stderr:
                    try:
                        stderr_line = await asyncio.wait_for(
                            connection.process.stderr.readline(),
                            timeout=1.0
                        )
                        
                        if stderr_line:
                            error_msg = stderr_line.decode().strip()
                            connection.error_messages.append(error_msg)
                            logger.warning("MCP server %s stderr: %s", server_id, error_msg)
                            
                            # Emit error event
                            await self._emit_event('server_error', {
                                'server_id': server_id,
                                'error': error_msg
                            })
                    
                    except asyncio.TimeoutError:
                        # No new stderr output
                        pass
                
                await asyncio.sleep(0.1)  # Check every 100ms
            
            # Connection has ended
            if connection.status == MCPConnectionStatus.CONNECTED:
                connection.status = MCPConnectionStatus.DISCONNECTED
                self._stats['active_connections'] -= 1
                
                logger.info("MCP server %s connection ended", server_id)
                await self._emit_event('server_disconnected', {
                    'server_id': server_id,
                    'config': connection.config
                })
        
        except Exception as e:
            logger.error("Error monitoring connection %s: %s", server_id, e)
            connection.status = MCPConnectionStatus.ERROR
            connection.error_messages.append(str(e))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get MCP Manager statistics"""
        stats = self._stats.copy()
        
        # Add connection-specific statistics
        connection_stats = {
            'total_servers': len(self._connections),
            'connected_servers': sum(1 for c in self._connections.values() if c.status == MCPConnectionStatus.CONNECTED),
            'error_servers': sum(1 for c in self._connections.values() if c.status == MCPConnectionStatus.ERROR),
            'healthy_servers': sum(1 for c in self._connections.values() if c.is_healthy())
        }
        
        stats.update(connection_stats)
        return stats
    
    def reset_statistics(self):
        """Reset all statistics"""
        self._stats = {
            'total_connections': 0,
            'active_connections': 0,
            'failed_connections': 0,
            'total_restarts': 0,
            'avg_connection_time': 0.0,
            'total_uptime': 0.0
        }
        
        # Reset connection-specific statistics
        for connection in self._connections.values():
            connection.connection_attempts = 0
            connection.error_messages.clear()
    
    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, '_initialized') and self._initialized:
            try:
                # Create a new event loop if needed
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule cleanup in the running loop
                    loop.create_task(self.stop())
                else:
                    # Run cleanup synchronously
                    loop.run_until_complete(self.stop())
            except Exception:
                pass  # Best effort cleanup


# Global MCP Manager instance
mcp_manager = MCPManager()

# Convenience functions
async def get_mcp_manager() -> MCPManager:
    """Get the global MCP Manager instance"""
    await mcp_manager.start()
    return mcp_manager


async def register_mcp_server(config: MCPServerConfig) -> str:
    """Register an MCP server using global manager"""
    manager = await get_mcp_manager()
    return await manager.register_server(config)


async def connect_mcp_server(server_id: str) -> bool:
    """Connect to MCP server using global manager"""
    manager = await get_mcp_manager()
    return await manager.connect(server_id)


def create_mcp_config(
    name: str,
    command: str,
    args: List[str] = None,
    **kwargs
) -> MCPServerConfig:
    """Create MCP server configuration"""
    args = args or []
    
    return MCPServerConfig(
        name=name,
        command=[command] + args,
        **kwargs
    )