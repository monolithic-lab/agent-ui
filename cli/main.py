# cli/main.py
import asyncio
import click
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

from cli.config import load_config, validate_config
from cli.commands import ChatCommand, ConfigCommand
from exceptions.base import ModelServiceError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentCLI(click.Group):
    """Main CLI application"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = None
        self.chat_cmd = None
        self.config_cmd = None
    
    def initialize(self, config_path: Optional[str] = None):
        """Initialize CLI with configuration"""
        try:
            # Load configuration
            self.config = load_config(config_path)
            validate_config(self.config)
            
            # Initialize commands
            self.chat_cmd = ChatCommand(self.config)
            self.config_cmd = ConfigCommand(self.config)
            
            logger.info("CLI initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CLI: {e}")
            raise click.ClickException(f"Configuration error: {e}")
    
    def run_chat(self, session_id: Optional[str], model: Optional[str], provider: Optional[str]):
        """Run interactive chat session"""
        if not self.chat_cmd:
            raise click.ClickException("CLI not initialized")
        
        try:
            asyncio.run(self.chat_cmd.interactive_chat(session_id, model, provider))
        except KeyboardInterrupt:
            click.echo("\nGoodbye!")
        except Exception as e:
            logger.error(f"Chat session failed: {e}")
            raise click.ClickException(f"Chat error: {e}")
    
    def run_config(self, action: str):
        """Run configuration management"""
        if not self.config_cmd:
            raise click.ClickException("CLI not initialized")
        
        try:
            if action == "show":
                self.config_cmd.show_config()
            elif action == "validate":
                self.config_cmd.validate_config()
            elif action == "setup":
                self.config_cmd.setup_interactive()
            else:
                raise click.ClickException(f"Unknown config action: {action}")
        except Exception as e:
            logger.error(f"Config command failed: {e}")
            raise click.ClickException(f"Config error: {e}")

# Create main CLI
@click.group(cls=AgentCLI)
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.pass_context
def cli(ctx, config, debug):
    """Multi-provider agent CLI with MCP integration"""
    # Setup debug logging
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize CLI
    ctx.ensure_object(dict)
    cli_obj = ctx.obj['cli'] = AgentCLI()
    cli_obj.initialize(config)

@cli.command()
@click.option('--session', '-s', help='Session ID to resume')
@click.option('--model', '-m', help='Model to use')
@click.option('--provider', '-p', type=click.Choice(['openai', 'anthropic', 'gemini', 'huggingface']), help='Provider to use')
@click.option('--non-interactive', is_flag=True, help='Run in non-interactive mode')
@click.argument('message', required=False)
@click.pass_context
def chat(ctx, session, model, provider, non_interactive, message):
    """Start a chat session"""
    cli_obj = ctx.obj['cli']
    
    if non_interactive and not message:
        click.echo("Message required for non-interactive mode", err=True)
        return
    
    cli_obj.run_chat(session, model, provider)

@cli.group()
def config():
    """Configuration management"""
    pass

@config.command('show')
@click.pass_context
def config_show(ctx):
    """Show current configuration"""
    ctx.obj['cli'].run_config('show')

@config.command('validate')
@click.pass_context
def config_validate(ctx):
    """Validate configuration"""
    ctx.obj['cli'].run_config('validate')

@config.command('setup')
@click.pass_context
def config_setup(ctx):
    """Interactive configuration setup"""
    ctx.obj['cli'].run_config('setup')

def main():
    """Main entry point"""
    cli()

if __name__ == '__main__':
    main()