# safety/loop_detection.py
import hashlib
import logging
import json
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict

from exceptions.base import LoopDetectionError

logger = logging.getLogger(__name__)

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