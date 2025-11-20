# cli/commands.py
"""
CLI command implementations
"""

import asyncio
import json
import click
from typing import Optional, Dict, Any
from pathlib import Path

from database import DatabaseManager
from agents.assistant import Assistant, AgentConfig

class ChatCommand:
    """Interactive chat command"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db = DatabaseManager(
            config.get('database', {}).get('connection_string'),
            config.get('database', {}).get('db_name')
        )
        self.current_session = None
        self.current_agent = None
    
    async def interactive_chat(
        self, 
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None
    ):
        """Start interactive chat session"""
        await self.db.connect()
        
        try:
            # Create or resume session
            if session_id:
                self.current_session = await self.db.get_session(session_id)
                if not self.current_session:
                    click.echo(f"Session {session_id} not found")
                    return
                click.echo(f"Resumed session: {self.current_session.title}")
            else:
                # Create new session
                title = click.prompt("Session title", default="New Chat")
                provider = provider or click.prompt("Provider", type=click.Choice(['openai', 'anthropic', 'gemini', 'huggingface']))
                model = model or click.prompt("Model", default="gpt-4")
                
                from datetime import datetime, timezone
                from database import Session
                
                self.current_session = Session(
                    id=f"session_{datetime.now().timestamp()}",
                    user_id="cli_user",
                    title=title,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                    model=model,
                    provider=provider,
                    settings={}
                )
                click.echo(f"Created new session: {title}")
            
            # Initialize agent
            await self._initialize_agent()
            
            # Chat loop
            await self._chat_loop()
            
        finally:
            await self.db.disconnect()
    
    async def _initialize_agent(self):
        """Initialize the chat agent"""
        # Create LLM (simplified for example)
        llm = await self._create_llm(
            provider=self.current_session.provider,
            model=self.current_session.model
        )
        
        # Create agent config
        config = AgentConfig(
            name="cli_assistant",
            description="CLI Assistant",
            system_message="You are a helpful assistant in a CLI environment.",
            llm=llm,
            max_iterations=10
        )
        
        # Create agent
        self.current_agent = Assistant(config)
    
    async def _chat_loop(self):
        """Main chat loop"""
        click.echo("\nEnter your message (or 'quit' to exit, 'help' for commands):")
        
        while True:
            try:
                # Get user input
                user_input = click.prompt("You", prompt_suffix=" ")
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    click.echo("Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                if user_input.lower() == 'clear':
                    click.clear()
                    continue
                
                # Process message
                await self._process_message(user_input)
                
            except KeyboardInterrupt:
                click.echo("\nGoodbye!")
                break
            except Exception as e:
                click.echo(f"Error: {e}")
    
    async def _process_message(self, message: str):
        """Process user message"""
        from llm.schema import Message
        
        # Create user message
        user_message = Message(role='user', content=message)
        
        # Get agent response
        click.echo("Assistant: ", nl=False)
        
        try:
            async for response in self.current_agent.run([user_message]):
                for msg in response:
                    click.echo(msg.content, nl=False)
                    click.echo()  # New line
            
            # Save to database
            await self.db.save_message(
                session_id=self.current_session.id,
                role='user',
                content=message
            )
        except Exception as e:
            click.echo(f"Error processing message: {e}")
    
    def _show_help(self):
        """Show help commands"""
        click.echo("\nAvailable commands:")
        click.echo("  quit, exit, q - Exit the chat")
        click.echo("  help - Show this help")
        click.echo("  clear - Clear the screen")
        click.echo("\nYou can also use regular conversation with the AI assistant.")
    
    async def _create_llm(self, provider: str, model: str) -> 'BaseChatModel':
        """Create LLM instance (simplified)"""
        # This would integrate with your provider.py classes
        from provider import ProviderFactory
        from llm.base_chat_model import BaseChatModel
        
        provider_instance = ProviderFactory.create_provider(provider, self.config)
        return BaseChatModel(provider_instance)

class ConfigCommand:
    """Configuration management commands"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def show_config(self):
        """Show current configuration"""
        click.echo("Current Configuration:")
        
        # Filter out API keys for security
        config_copy = self.config.copy()
        for provider in config_copy.get('providers', {}):
            if 'api_key' in config_copy['providers'][provider]:
                config_copy['providers'][provider]['api_key'] = '***HIDDEN***'
        
        click.echo(json.dumps(config_copy, indent=2))
    
    def validate_config(self):
        """Validate configuration"""
        try:
            from cli.config import validate_config
            validate_config(self.config)
            click.echo("Configuration is valid")
        except Exception as e:
            click.echo(f"Configuration validation failed: {e}")
    
    def setup_interactive(self):
        """Interactive configuration setup"""
        click.echo("Interactive Configuration Setup")
        
        # Database setup
        self.config['database'] = {
            'connection_string': click.prompt("MongoDB URL", default="mongodb://localhost:27017"),
            'db_name': click.prompt("Database name", default="agent_ui")
        }
        
        # Provider setup
        providers = {}
        for provider in ['openai', 'anthropic', 'gemini']:
            if click.confirm(f"Setup {provider}?"):
                providers[provider] = {
                    'api_key': click.prompt(f"{provider} API key", hide_input=True),
                    'model': click.prompt(f"{provider} default model", default="")
                }
        
        self.config['providers'] = providers
        
        # Save config
        config_path = Path("config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        click.echo(f"Configuration saved to {config_path}")