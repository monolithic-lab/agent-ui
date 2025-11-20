# cli/config.py
"""
Configuration management for CLI
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file or environment"""
    config = {}
    
    # Load from file if specified
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config.update(json.load(f))
    
    # Load from .env file if exists
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    
    # Override with environment variables
    env_mappings = {
        'MONGODB_URL': ('database', 'connection_string'),
        'DB_NAME': ('database', 'db_name'),
        'OPENAI_API_KEY': ('providers', 'openai', 'api_key'),
        'ANTHROPIC_API_KEY': ('providers', 'anthropic', 'api_key'),
        'GOOGLE_API_KEY': ('providers', 'gemini', 'api_key'),
        'HUGGINGFACE_API_TOKEN': ('providers', 'huggingface', 'api_key'),
    }
    
    for env_var, config_path in env_mappings.items():
        value = os.getenv(env_var)
        if value:
            # Set nested value
            current = config
            for key in config_path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[config_path[-1]] = value
    
    # Default values
    config.setdefault('database', {
        'connection_string': 'mongodb://localhost:27017',
        'db_name': 'agent_ui'
    })
    
    config.setdefault('providers', {})
    
    return config

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration"""
    # Check required database config
    if 'database' not in config:
        raise ValueError("Database configuration is required")
    
    db_config = config['database']
    if 'connection_string' not in db_config:
        raise ValueError("Database connection_string is required")
    
    # Check provider configurations
    for provider_name, provider_config in config.get('providers', {}).items():
        if 'api_key' not in provider_config:
            click.echo(f"Warning: No API key for provider '{provider_name}'")
    
    return True