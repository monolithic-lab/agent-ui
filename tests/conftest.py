# tests/conftest.py
"""
Test configuration and fixtures
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import Generator

from database import DatabaseManager, Session, Message
from provider import BaseProvider, ModelResponse
from agents.base_agent import AgentConfig

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def mock_db():
    """Mock database manager"""
    db = Mock(spec=DatabaseManager)
    db.connect = AsyncMock()
    db.disconnect = AsyncMock()
    db.save_message = AsyncMock()
    db.create_session = AsyncMock()
    db.get_session = AsyncMock()
    return db

@pytest.fixture
def mock_provider():
    """Mock provider"""
    provider = Mock(spec=BaseProvider)
    provider.create_chat_completion = AsyncMock()
    provider._format_tools = Mock(return_value=[])
    provider._parse_response = Mock(return_value=ModelResponse(content="Mock response"))
    provider._get_provider_info = Mock(return_value=Mock(
        name="mock",
        supports_streaming=True,
        supports_thinking=False,
        supports_tools=True,
        supports_multimodal=False
    ))
    return provider

@pytest.fixture
def sample_agent_config():
    """Sample agent configuration"""
    return AgentConfig(
        name="test_agent",
        description="Test agent",
        system_message="You are a test assistant.",
        max_iterations=5,
        temperature=0.7
    )

@pytest.fixture
def sample_messages():
    """Sample messages for testing"""
    from llm.schema import Message
    return [
        Message(role='user', content='Hello'),
        Message(role='assistant', content='Hi there!'),
    ]

@pytest.fixture
def sample_tool_schema():
    """Sample tool schema"""
    return {
        'name': 'test_tool',
        'description': 'A test tool',
        'parameters': {
            'type': 'object',
            'properties': {
                'input': {'type': 'string'}
            },
            'required': ['input']
        }
    }