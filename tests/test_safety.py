# tests/test_safety.py
"""
Test safety and loop detection
"""

import pytest
from safety.loop_detection import LoopDetector, loop_detector, detect_loop
from exceptions.base import LoopDetectionError

class TestLoopDetector:
    """Test loop detection functionality"""
    
    def test_loop_detection_initialization(self):
        """Test loop detector initialization"""
        detector = LoopDetector(max_tool_iterations=5, max_content_repeats=3)
        assert detector.max_tool_iterations == 5
        assert detector.max_content_repeats == 3
        assert detector.session_id is None
        assert detector.iteration_count == 0
    
    def test_session_start(self):
        """Test session management"""
        detector = LoopDetector()
        detector.start_session("test_session")
        
        assert detector.session_id == "test_session"
        assert detector.iteration_count == 0
        assert len(detector.tool_call_history) == 0
        assert len(detector.tool_call_counts) == 0
    
    def test_tool_call_limit_exceeded(self):
        """Test tool call iteration limit"""
        detector = LoopDetector(max_tool_iterations=2)
        detector.start_session("test")
        
        # Should allow up to max_tool_iterations
        detector.check_tool_call("tool1", {})
        
        with pytest.raises(LoopDetectionError) as exc_info:
            detector.check_tool_call("tool2", {})
            detector.check_tool_call("tool3", {})  # This should exceed limit
        
        assert "Maximum iterations" in str(exc_info.value)
    
    def test_excessive_tool_repetition(self):
        """Test detection of excessive tool repetition"""
        detector = LoopDetector(max_tool_iterations=10, max_content_repeats=2)
        detector.start_session("test")
        
        # Should allow up to max_content_repeats calls
        detector.check_tool_call("tool1", {})
        detector.check_tool_call("tool1", {})
        
        with pytest.raises(LoopDetectionError) as exc_info:
            detector.check_tool_call("tool1", {})  # Third call should exceed limit
        
        assert "repeated" in str(exc_info.value)
    
    def test_content_repetition_detection(self):
        """Test content repetition detection"""
        detector = LoopDetector()
        detector.start_session("test")
        
        # First content should pass
        detector.check_content("Hello world")
        
        # Second unique content should pass
        detector.check_content("Goodbye world")
        
        # Repeated content should fail
        with pytest.raises(LoopDetectionError) as exc_info:
            detector.check_content("Hello world")
        
        assert "repeated" in str(exc_info.value)
    
    def test_different_tool_arguments(self):
        """Test that different tool arguments don't trigger repetition"""
        detector = LoopDetector(max_content_repeats=2)
        detector.start_session("test")
        
        # Same tool with different args should be allowed
        detector.check_tool_call("tool1", {"arg": "value1"})
        detector.check_tool_call("tool1", {"arg": "value2"})
        
        # Same tool with same args should be limited
        detector.check_tool_call("tool1", {"arg": "value1"})
        
        with pytest.raises(LoopDetectionError):
            detector.check_tool_call("tool1", {"arg": "value1"})  # Fourth call

class TestGlobalLoopDetection:
    """Test global loop detection function"""
    
    def test_global_detect_loop(self):
        """Test global loop detection function"""
        # Reset global detector
        loop_detector.start_session("global_test")
        
        # Should work normally
        detect_loop("tool1", {"arg": "value1"}, "content1", "global_test")
        
        # Should detect iteration limit
        for i in range(loop_detector.max_tool_iterations + 1):
            try:
                detect_loop("tool2", {"arg": f"value{i}"}, None, "global_test")
            except LoopDetectionError:
                break
        
        # Should have triggered the limit
        assert loop_detector.iteration_count > loop_detector.max_tool_iterations
    
    def test_session_isolation(self):
        """Test that different sessions are isolated"""
        # Start first session
        loop_detector.start_session("session1")
        loop_detector.check_tool_call("tool1", {})
        assert loop_detector.session_id == "session1"
        
        # Start second session - should reset state
        loop_detector.start_session("session2")
        assert loop_detector.iteration_count == 0
        assert loop_detector.session_id == "session2"
    
    def test_get_stats(self):
        """Test statistics collection"""
        detector = LoopDetector()
        detector.start_session("test_stats")
        
        detector.check_tool_call("tool1", {"arg": "value1"})
        detector.check_tool_call("tool2", {"arg": "value2"})
        detector.check_content("test content")
        
        stats = detector.get_stats()
        
        assert stats['session_id'] == "test_stats"
        assert stats['iteration_count'] == 2
        assert stats['unique_tool_calls'] == 2
        assert stats['tool_call_history_length'] == 2
        assert stats['content_hashes_count'] == 1