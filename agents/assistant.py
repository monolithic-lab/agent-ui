# agents/assistant.py
import logging
from typing import List, Dict, Any, AsyncIterator, Optional

from agents.fncall_agent import FnCallAgent
from agents.base_agent import AgentConfig
from llm.schema import Message

logger = logging.getLogger(__name__)

class Assistant(FnCallAgent):
    """Main user-facing assistant agent"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        # Assistant-specific initialization
        self.rag_enabled = config.rag_enabled if hasattr(config, 'rag_enabled') else False
        self.memory_enabled = config.memory_enabled if hasattr(config, 'memory_enabled') else False
    
    async def _run(
        self, 
        messages: List[Message], 
        **kwargs
    ) -> AsyncIterator[List[Message]]:
        """Main assistant execution loop"""
        
        # Pre-process messages (add system context, RAG, etc.)
        processed_messages = await self._preprocess_messages(messages)
        
        # Use parent function calling logic
        async for response in super()._run(processed_messages, **kwargs):
            yield response
    
    async def _preprocess_messages(self, messages: List[Message]) -> List[Message]:
        """Pre-process messages for RAG, memory, etc."""
        processed_messages = []
        
        # Add system message
        processed_messages.extend(self._system_messages)
        
        # Add memory context if enabled
        if self.memory_enabled:
            memory_context = await self._get_memory_context(messages)
            if memory_context:
                processed_messages.append(memory_context)
        
        # Add RAG context if enabled
        if self.rag_enabled:
            rag_context = await self._get_rag_context(messages)
            if rag_context:
                processed_messages.append(rag_context)
        
        # Add user messages
        processed_messages.extend(messages)
        
        return processed_messages
    
    async def _get_memory_context(self, messages: List[Message]) -> Optional[Message]:
        """Get memory context from previous conversations"""
        # Implementation would fetch relevant memory from database
        # For now, return empty context
        return None
    
    async def _get_rag_context(self, messages: List[Message]) -> Optional[Message]:
        """Get RAG context from documents"""
        # Implementation would search documents and retrieve relevant context
        # For now, return empty context
        return None

# Register the assistant
from agents.base_agent import AgentRegistry

@AgentRegistry.register('assistant')
class RegisteredAssistant(Assistant):
    pass