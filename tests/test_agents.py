# tests/test_agents.py
"""
Test agent system functionality
"""

import pytest
from unittest.mock import Mock, AsyncMock

from agents.base_agent import BaseAgent, AgentConfig, AgentRegistry
from agents.fncall_agent import FnCallAgent
from agents.assistant import Assistant
from tools.base_tool import BaseTool, ToolResult
from llm.schema import Message, ASSISTANT, FunctionCall
from exceptions.base import ModelServiceError

class TestAgentConfig:
    """Test AgentConfig functionality"""
    
    def test_agent_config_creation(self):
        """Test agent config creation"""
        config = AgentConfig(
            name="test_agent",
            description="A test agent",
            system_message="You are a test assistant.",
            max_iterations=10
        )
        
        assert config.name == "test_agent"
        assert config.max_iterations == 10
        assert config.temperature == 0.7
        assert not config.rag_enabled
        assert not config.memory_enabled

class TestAgentRegistry:
    """Test AgentRegistry functionality"""
    
    def test_register_agent(self):
        """Test agent registration"""
        # Clear registry
        AgentRegistry._agents.clear()
        
        @AgentRegistry.register('test_agent')
        class TestAgent(BaseAgent):
            async def _run(self, messages, **kwargs):
                yield []
        
        assert 'test_agent' in AgentRegistry._agents
        assert AgentRegistry._agents['test_agent'] == TestAgent
    
    def test_create_agent(self):
        """Test agent creation from registry"""
        # Clear registry
        AgentRegistry._agents.clear()
        
        @AgentRegistry.register('test_agent')
        class TestAgent(BaseAgent):
            async def _run(self, messages, **kwargs):
                yield []
        
        config = AgentConfig(name="test", description="test", system_message="test")
        agent = AgentRegistry.create_agent('test_agent', config)
        
        assert isinstance(agent, TestAgent)
        assert agent.config.name == "test"

class TestFnCallAgent:
    """Test FnCallAgent functionality"""
    
    @pytest.fixture
    def mock_llm(self):
        llm = Mock()
        llm.chat_with_retry = AsyncMock()
        llm._handle_provider_error = Mock(return_value=ModelServiceError(message="Test error"))
        return llm
    
    @pytest.fixture
    def mock_tool(self):
        tool = Mock(spec=BaseTool)
        tool.enabled = True
        tool.name = "test_tool"
        tool.get_schema.return_value = Mock(
            name="test_tool",
            description="A test tool",
            parameters={},
            required=[]
        )
        tool.safe_execute = AsyncMock(return_value=ToolResult(content="Tool result"))
        return tool
    
    @pytest.mark.asyncio
    async def test_simple_response(self, mock_llm, mock_tool):
        """Test agent response without function calls"""
        config = AgentConfig(
            name="test",
            description="test",
            system_message="test",
            llm=mock_llm,
            tools=[mock_tool]
        )
        
        agent = FnCallAgent(config)
        
        # Mock LLM response with content
        mock_llm.chat_with_retry.return_value = [
            Message(role=ASSISTANT, content="Hello!")
        ]
        
        messages = [Message(role='user', content="Hi")]
        results = []
        async for response in agent.run(messages):
            results.extend(response)
        
        assert len(results) > 0
        assert results[0].content == "Hello!"
    
    @pytest.mark.asyncio
    async def test_function_calling(self, mock_llm, mock_tool):
        """Test agent function calling"""
        config = AgentConfig(
            name="test",
            description="test",
            system_message="test",
            llm=mock_llm,
            tools=[mock_tool]
        )
    
        agent = FnCallAgent(config)
    
        # Mock LLM response with function call
        function_call = FunctionCall(
            id="call_1",
            name="test_tool",
            arguments='{"input": "test"}'
        )
    
        mock_llm.chat_with_retry.side_effect = [
            [Message(role=ASSISTANT, content="", tool_calls=[function_call])],  # Add empty content
            [Message(role=ASSISTANT, content="Tool executed successfully")]
        ]
    
    @pytest.mark.asyncio
    async def test_assistant_creation(self, mock_llm):
        """Test assistant agent creation"""
        config = AgentConfig(
            name="assistant",
            description="Test assistant",
            system_message="You are a test assistant.",
            llm=mock_llm
        )
        
        assistant = Assistant(config)
        assert isinstance(assistant, FnCallAgent)
        assert assistant.config.name == "assistant"