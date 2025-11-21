# utils/safety.py
"""
Safety module combining loop detection and safety features
Advanced Loop Detection System
Prevents infinite loops and repetitive behavior in agent conversations
"""

import asyncio
import hashlib
import logging
import time
import json
from collections import Counter, defaultdict, deque
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from core.exceptions import LoopDetectionError

logger = logging.getLogger(__name__)

class LoopType(Enum):
    """Types of loops that can be detected"""
    TOOL_REPETITION = "tool_repetition"
    CONTENT_LOOP = "content_loop"
    AGENT_IDLE = "agent_idle"
    CONTEXT_OVERFLOW = "context_overflow"
    CYCLE_DETECTION = "cycle_detection"


@dataclass
class LoopDetectionConfig:
    """Configuration for loop detection"""
    # Tool call thresholds
    max_tool_repetitions: int = 5
    max_consecutive_tool_calls: int = 10
    
    # Content analysis thresholds
    max_content_repetitions: int = 10
    content_similarity_threshold: float = 0.85
    
    # Time-based detection
    max_idle_time_seconds: int = 300  # 5 minutes
    max_conversation_time_seconds: int = 3600  # 1 hour
    
    # Context management
    max_context_length: int = 100000  # token limit
    context_compression_threshold: int = 50000
    
    # LLM-based detection
    enable_llm_detection: bool = True
    max_llm_analysis_calls: int = 3
    
    # Performance thresholds
    max_memory_mb: int = 500
    max_processing_time_seconds: int = 30


# Basic loop detection functionality
class LoopDetector:
    """Detect and prevent infinite loops in tool execution"""
    
    def __init__(self, max_tool_iterations: int = 10, max_content_repeats: int = 5):
        self.max_tool_iterations = max_tool_iterations
        self.max_content_repeats = max_content_repeats
        
        # Track tool call patterns
        self.tool_call_history: List[str] = []
        self.tool_call_counts: Dict[str, int] = defaultdict(int)
        
        # Track content patterns - FIXED: Changed from list to set
        self.content_hashes: Set[str] = set()
        
        # Session tracking
        self.session_id: Optional[str] = None
        self.iteration_count = 0
    
    def start_session(self, session_id: str) -> None:
        """Start a new detection session"""
        self.session_id = session_id
        self.tool_call_history.clear()
        self.tool_call_counts.clear()
        self.content_hashes.clear()
        self.iteration_count = 0
        
        logger.info(f"Started loop detection session: {session_id}")
    
    def check_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Check if tool call pattern indicates a loop"""
        self.iteration_count += 1
        
        # Check iteration limit
        if self.iteration_count > self.max_tool_iterations:
            raise LoopDetectionError(
                message=f"Maximum iterations ({self.max_tool_iterations}) exceeded"
            )
        
        # Create tool call signature
        args_hash = hashlib.md5(
            json.dumps(arguments, sort_keys=True).encode()
        ).hexdigest() if arguments else ""
        
        tool_signature = f"{tool_name}:{args_hash}"
        
        # Check for repeated patterns
        self.tool_call_history.append(tool_signature)
        self.tool_call_counts[tool_signature] += 1
        
        # Remove old entries
        if len(self.tool_call_history) > self.max_tool_iterations * 2:
            self.tool_call_history.pop(0)
        
        # Check for excessive repetition
        if self.tool_call_counts[tool_signature] > self.max_content_repeats:
            raise LoopDetectionError(
                message=f"Tool call '{tool_name}' repeated {self.tool_call_counts[tool_signature]} times"
            )
    
    def check_content(self, content: str) -> None:
        """Check if content is being repeated"""
        if not content:
            return
        
        # Hash content
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        if content_hash in self.content_hashes:
            raise LoopDetectionError(
                message="Content is being repeated"
            )
        
        self.content_hashes.add(content_hash)
        
        # Limit history size - FIXED: Proper set handling
        if len(self.content_hashes) > 100:
            # Convert to list, remove oldest, convert back to set
            hashes_list = list(self.content_hashes)
            self.content_hashes = set(hashes_list[50:])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        most_common_tool = None
        if self.tool_call_counts:
            most_common_tool = max(self.tool_call_counts.items(), key=lambda x: x[1])[0]
        
        return {
            'session_id': self.session_id,
            'iteration_count': self.iteration_count,
            'unique_tool_calls': len(self.tool_call_counts),
            'tool_call_history_length': len(self.tool_call_history),
            'content_hashes_count': len(self.content_hashes),
            'most_common_tool': most_common_tool
        }


# Global loop detector instance
loop_detector = LoopDetector()


def detect_loop(
    tool_name: str, 
    arguments: Dict[str, Any], 
    content: Optional[str] = None,
    session_id: str = "default"
) -> None:
    """Detect loops in agent execution"""
    if loop_detector.session_id != session_id:
        loop_detector.start_session(session_id)
    
    loop_detector.check_tool_call(tool_name, arguments)
    if content:
        loop_detector.check_content(content)


# Advanced loop detection service
class LoopDetectionService:
    """Implements multiple detection strategies with configurable thresholds"""
    
    def __init__(self, config: LoopDetectionConfig = None):
        self.config = config or LoopDetectionConfig()
        
        # Detection state
        self._tool_call_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._content_hashes: deque = deque(maxlen=100)
        self._agent_states: Dict[str, Dict[str, Any]] = {}
        self._conversation_start_time: Dict[str, float] = {}
        self._last_activity_time: Dict[str, float] = {}
        
        # Statistics
        self._detection_stats: Dict[str, int] = defaultdict(int)
        self._false_positives: Set[str] = set()
        
        # Performance tracking
        self._processing_times: List[float] = []
        
        logger.info("LoopDetectionService initialized with config: %s", self.config)
    
    async def analyze_conversation(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]],
        agent_name: str = "default"
    ) -> Dict[str, Any]:
        """
        Analyze conversation for loop patterns
        
        Args:
            conversation_id: Unique conversation identifier
            messages: List of messages in conversation
            agent_name: Name of the agent processing the conversation
        
        Returns:
            Analysis results with detected loops and recommendations
        """
        start_time = time.time()
        
        try:
            # Initialize conversation state if new
            self._initialize_conversation_state(conversation_id, agent_name)
            
            # Perform multiple detection strategies
            results = {
                'conversation_id': conversation_id,
                'analysis_time': start_time,
                'detected_loops': [],
                'risk_level': 'low',
                'recommendations': [],
                'statistics': {}
            }
            
            # 1. Tool call repetition detection
            tool_results = await self._detect_tool_repetitions(
                conversation_id, messages
            )
            results['detected_loops'].extend(tool_results)
            
            # 2. Content similarity detection
            content_results = await self._detect_content_loops(
                conversation_id, messages
            )
            results['detected_loops'].extend(content_results)
            
            # 3. Agent idle detection
            idle_results = await self._detect_agent_idle(
                conversation_id, agent_name
            )
            results['detected_loops'].extend(idle_results)
            
            # 4. Context overflow detection
            context_results = await self._detect_context_overflow(
                conversation_id, messages
            )
            results['detected_loops'].extend(context_results)
            
            # 5. LLM-based cycle detection (if enabled)
            if self.config.enable_llm_detection:
                llm_results = await self._detect_llm_cycles(
                    conversation_id, messages
                )
                results['detected_loops'].extend(llm_results)
            
            # Determine overall risk level
            results['risk_level'] = self._calculate_risk_level(results['detected_loops'])
            
            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(results['detected_loops'])
            
            # Add statistics
            results['statistics'] = self._get_detection_statistics()
            
            # Track processing time
            processing_time = time.time() - start_time
            self._processing_times.append(processing_time)
            
            # Log results
            if results['detected_loops']:
                logger.warning(
                    "Loop detection found %d issues for conversation %s: %s",
                    len(results['detected_loops']),
                    conversation_id,
                    [loop['type'] for loop in results['detected_loops']]
                )
            else:
                logger.debug("No loops detected for conversation %s", conversation_id)
            
            return results
            
        except Exception as e:
            logger.error("Loop detection failed: %s", e)
            return {
                'conversation_id': conversation_id,
                'error': str(e),
                'detected_loops': [],
                'risk_level': 'unknown'
            }
    
    async def _detect_tool_repetitions(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect repetitive tool calls"""
        loops = []
        
        # Extract tool calls from messages
        tool_calls = []
        for message in messages:
            if message.get('role') == 'assistant' and 'tool_calls' in message:
                for tool_call in message['tool_calls']:
                    tool_name = tool_call.get('function', {}).get('name', 'unknown')
                    tool_calls.append(tool_name)
        
        # Check for excessive tool call repetitions
        if tool_calls:
            tool_counts = Counter(tool_calls)
            for tool_name, count in tool_counts.items():
                if count > self.config.max_tool_repetitions:
                    loops.append({
                        'type': LoopType.TOOL_REPETITION.value,
                        'severity': 'high',
                        'description': f"Tool '{tool_name}' called {count} times",
                        'count': count,
                        'threshold': self.config.max_tool_repetitions,
                        'tool_name': tool_name,
                        'recommendation': 'Implement tool result caching or agent memory'
                    })
        
        # Check for consecutive tool calls
        consecutive_count = 0
        for tool_call in tool_calls:
            if tool_call != 'tool_result':  # Exclude tool results
                consecutive_count += 1
            else:
                consecutive_count = 0
            
            if consecutive_count > self.config.max_consecutive_tool_calls:
                loops.append({
                    'type': LoopType.TOOL_REPETITION.value,
                    'severity': 'medium',
                    'description': f"Excessive consecutive tool calls: {consecutive_count}",
                    'count': consecutive_count,
                    'threshold': self.config.max_consecutive_tool_calls,
                    'recommendation': 'Review agent logic for unnecessary tool calls'
                })
                break
        
        return loops
    
    async def _detect_content_loops(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect content similarity and repetition"""
        loops = []
        
        # Extract content from messages
        contents = []
        for message in messages:
            if message.get('role') in ['assistant', 'user']:
                content = str(message.get('content', '')).strip()
                if content:
                    contents.append(content)
        
        # Calculate content similarity
        if len(contents) > 1:
            similarity_groups = self._find_similar_content(contents)
            
            for group in similarity_groups:
                if len(group) >= self.config.max_content_repetitions:
                    loops.append({
                        'type': LoopType.CONTENT_LOOP.value,
                        'severity': 'medium',
                        'description': f"Similar content repeated {len(group)} times",
                        'count': len(group),
                        'threshold': self.config.max_content_repetitions,
                        'similarity_groups': group,
                        'recommendation': 'Implement content deduplication'
                    })
        
        return loops
    
    async def _detect_agent_idle(
        self,
        conversation_id: str,
        agent_name: str
    ) -> List[Dict[str, Any]]:
        """Detect agent idle behavior"""
        loops = []
        
        current_time = time.time()
        
        # Check idle time
        if conversation_id in self._last_activity_time:
            idle_time = current_time - self._last_activity_time[conversation_id]
            if idle_time > self.config.max_idle_time_seconds:
                loops.append({
                    'type': LoopType.AGENT_IDLE.value,
                    'severity': 'medium',
                    'description': f"Agent idle for {idle_time:.0f} seconds",
                    'idle_time': idle_time,
                    'threshold': self.config.max_idle_time_seconds,
                    'agent_name': agent_name,
                    'recommendation': 'Implement timeout handling'
                })
        
        # Check total conversation time
        if conversation_id in self._conversation_start_time:
            conversation_time = current_time - self._conversation_start_time[conversation_id]
            if conversation_time > self.config.max_conversation_time_seconds:
                loops.append({
                    'type': LoopType.AGENT_IDLE.value,
                    'severity': 'high',
                    'description': f"Conversation running for {conversation_time:.0f} seconds",
                    'conversation_time': conversation_time,
                    'threshold': self.config.max_conversation_time_seconds,
                    'recommendation': 'Implement conversation timeout'
                })
        
        return loops
    
    async def _detect_context_overflow(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect context length overflow"""
        loops = []
        
        # Calculate approximate token count
        total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
        estimated_tokens = total_chars // 4  # Rough approximation
        
        if estimated_tokens > self.config.max_context_length:
            loops.append({
                'type': LoopType.CONTEXT_OVERFLOW.value,
                'severity': 'high',
                'description': f"Context length exceeded: {estimated_tokens} tokens",
                'current_length': estimated_tokens,
                'threshold': self.config.max_context_length,
                'recommendation': 'Implement context compression or summarization'
            })
        
        return loops
    
    async def _detect_llm_cycles(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """LLM-based cycle detection (placeholder for future implementation)"""
        # This would use LLM analysis to detect semantic loops
        # For now, return empty list
        return []
    
    def _find_similar_content(self, contents: List[str]) -> List[List[int]]:
        """Find groups of similar content using hashing"""
        content_hashes = []
        groups = []
        
        for content in contents:
            # Create content hash
            content_hash = hashlib.md5(content.lower().encode()).hexdigest()
            content_hashes.append(content_hash)
        
        # Find duplicate hashes
        hash_groups = defaultdict(list)
        for i, content_hash in enumerate(content_hashes):
            hash_groups[content_hash].append(i)
        
        # Return groups with more than threshold repetitions
        similar_groups = [
            [contents[i] for i in indices]
            for indices in hash_groups.values()
            if len(indices) > 1
        ]
        
        return similar_groups
    
    def _calculate_risk_level(self, detected_loops: List[Dict[str, Any]]) -> str:
        """Calculate overall risk level based on detected loops"""
        if not detected_loops:
            return 'low'
        
        high_severity_count = sum(1 for loop in detected_loops if loop.get('severity') == 'high')
        medium_severity_count = sum(1 for loop in detected_loops if loop.get('severity') == 'medium')
        
        if high_severity_count > 0:
            return 'high'
        elif medium_severity_count >= 2:
            return 'high'
        elif medium_severity_count > 0:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(self, detected_loops: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Collect unique recommendations
        for loop in detected_loops:
            if 'recommendation' in loop:
                recommendations.append(loop['recommendation'])
        
        # Add general recommendations
        if any(loop['type'] == LoopType.TOOL_REPETITION.value for loop in detected_loops):
            recommendations.append("Consider implementing tool result caching")
        
        if any(loop['type'] == LoopType.CONTEXT_OVERFLOW.value for loop in detected_loops):
            recommendations.append("Implement context compression or summarization")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _get_detection_statistics(self) -> Dict[str, Any]:
        """Get loop detection statistics"""
        return {
            'total_detections': sum(self._detection_stats.values()),
            'detection_types': dict(self._detection_stats),
            'false_positives': len(self._false_positives),
            'avg_processing_time': sum(self._processing_times) / max(len(self._processing_times), 1),
            'config': {
                'max_tool_repetitions': self.config.max_tool_repetitions,
                'max_content_repetitions': self.config.max_content_repetitions,
                'max_idle_time_seconds': self.config.max_idle_time_seconds,
                'enable_llm_detection': self.config.enable_llm_detection
            }
        }
    
    def _initialize_conversation_state(self, conversation_id: str, agent_name: str):
        """Initialize state for new conversation"""
        current_time = time.time()
        
        if conversation_id not in self._conversation_start_time:
            self._conversation_start_time[conversation_id] = current_time
        
        if conversation_id not in self._last_activity_time:
            self._last_activity_time[conversation_id] = current_time
        
        if agent_name not in self._agent_states:
            self._agent_states[agent_name] = {}
    
    def update_activity(self, conversation_id: str):
        """Update last activity time for conversation"""
        self._last_activity_time[conversation_id] = time.time()
    
    def mark_false_positive(self, conversation_id: str):
        """Mark detection as false positive for learning"""
        self._false_positives.add(conversation_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return self._get_detection_statistics()
    
    def reset_stats(self):
        """Reset all statistics"""
        self._detection_stats.clear()
        self._false_positives.clear()
        self._processing_times.clear()


# Global loop detection service instance
loop_detection_service = LoopDetectionService()

# Convenience functions
async def analyze_conversation_for_loops(
    conversation_id: str,
    messages: List[Dict[str, Any]],
    agent_name: str = "default"
) -> Dict[str, Any]:
    """Analyze conversation for loops using global service"""
    return await loop_detection_service.analyze_conversation(
        conversation_id, messages, agent_name
    )


def create_loop_detection_service(config: LoopDetectionConfig = None) -> LoopDetectionService:
    """Create a new loop detection service with custom config"""
    return LoopDetectionService(config)
