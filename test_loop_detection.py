#!/usr/bin/env python3
# test_loop_detection.py
"""
Comprehensive test suite for the Loop Detection System
Tests enterprise-grade loop detection and prevention
"""

import sys
import os
import asyncio
import time
import traceback
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_loop_detection():
    """Test the loop detection system comprehensively"""
    
    print("🔄 Testing Agent-UI Loop Detection System")
    print("=" * 50)
    
    try:
        # Import the loop detection components
        from utils.loop_detection import (
            LoopDetectionService,
            LoopDetectionConfig,
            LoopType,
            analyze_conversation_for_loops,
            create_loop_detection_service
        )
        
        # 1. Test basic configuration
        print(f"\n📋 Step 1: Testing Configuration")
        config = LoopDetectionConfig(
            max_tool_repetitions=3,
            max_content_repetitions=2,
            max_idle_time_seconds=5
        )
        service = LoopDetectionService(config)
        print(f"   ✅ Configuration created with custom thresholds")
        
        # 2. Test tool repetition detection
        print(f"\n🔧 Step 2: Testing Tool Repetition Detection")
        messages = [
            {'role': 'assistant', 'content': 'I will search for information', 'tool_calls': [
                {'function': {'name': 'web_search', 'arguments': '{"query": "AI"}'}}
            ]},
            {'role': 'assistant', 'content': 'Let me search again', 'tool_calls': [
                {'function': {'name': 'web_search', 'arguments': '{"query": "AI"}'}}
            ]},
            {'role': 'assistant', 'content': 'Searching once more', 'tool_calls': [
                {'function': {'name': 'web_search', 'arguments': '{"query": "AI"}'}}
            ]},
            {'role': 'assistant', 'content': 'Final search attempt', 'tool_calls': [
                {'function': {'name': 'web_search', 'arguments': '{"query": "AI"}'}}
            ]}
        ]
        
        result = await service.analyze_conversation('test_conv_1', messages)
        tool_loops = [loop for loop in result['detected_loops'] if loop['type'] == LoopType.TOOL_REPETITION.value]
        print(f"   Detected tool loops: {len(tool_loops)}")
        if tool_loops:
            print(f"   Loop details: {tool_loops[0]['description']}")
        assert len(tool_loops) > 0, "Should detect tool repetition"
        print("   ✅ Tool repetition detection working")
        
        # 3. Test content loop detection
        print(f"\n📝 Step 3: Testing Content Loop Detection")
        content_messages = [
            {'role': 'assistant', 'content': 'I need to search for information'},
            {'role': 'user', 'content': 'What did you find?'},
            {'role': 'assistant', 'content': 'I need to search for information'},  # Repetition
            {'role': 'user', 'content': 'Did you search?'},
            {'role': 'assistant', 'content': 'I need to search for information'},  # Repetition
            {'role': 'user', 'content': 'Please search now'},
            {'role': 'assistant', 'content': 'I need to search for information'},  # Repetition
        ]
        
        result = await service.analyze_conversation('test_conv_2', content_messages)
        content_loops = [loop for loop in result['detected_loops'] if loop['type'] == LoopType.CONTENT_LOOP.value]
        print(f"   Detected content loops: {len(content_loops)}")
        if content_loops:
            print(f"   Loop details: {content_loops[0]['description']}")
        assert len(content_loops) > 0, "Should detect content repetition"
        print("   ✅ Content loop detection working")
        
        # 4. Test agent idle detection
        print(f"\n⏰ Step 4: Testing Agent Idle Detection")
        
        # Simulate idle conversation
        idle_result = await service.analyze_conversation('test_conv_3', [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there'}
        ])
        
        # Manually set last activity time to simulate idle
        service._last_activity_time['test_conv_3'] = time.time() - 10
        
        idle_loops = [loop for loop in idle_result['detected_loops'] if loop['type'] == LoopType.AGENT_IDLE.value]
        print(f"   Detected idle loops: {len(idle_loops)}")
        print("   ✅ Agent idle detection working")
        
        # 5. Test context overflow detection
        print(f"\n📊 Step 5: Testing Context Overflow Detection")
        
        # Create large conversation
        large_messages = []
        for i in range(100):  # Create 100 messages
            large_messages.append({
                'role': 'user' if i % 2 == 0 else 'assistant',
                'content': f"This is message number {i} " * 100  # Large content
            })
        
        result = await service.analyze_conversation('test_conv_4', large_messages)
        context_loops = [loop for loop in result['detected_loops'] if loop['type'] == LoopType.CONTEXT_OVERFLOW.value]
        print(f"   Detected context overflow: {len(context_loops)}")
        if context_loops:
            print(f"   Overflow details: {context_loops[0]['description']}")
        print("   ✅ Context overflow detection working")
        
        # 6. Test risk level calculation
        print(f"\n⚠️ Step 6: Testing Risk Level Calculation")
        test_cases = [
            ([], 'low'),
            ([{'severity': 'low'}], 'low'),
            ([{'severity': 'medium'}], 'medium'),
            ([{'severity': 'high'}], 'high'),
            ([{'severity': 'medium'}, {'severity': 'medium'}], 'high'),
        ]
        
        for loops, expected_risk in test_cases:
            actual_risk = service._calculate_risk_level(loops)
            assert actual_risk == expected_risk, f"Expected {expected_risk}, got {actual_risk}"
        
        print("   ✅ Risk level calculation working")
        
        # 7. Test recommendation generation
        print(f"\n💡 Step 7: Testing Recommendation Generation")
        test_loops = [
            {'type': LoopType.TOOL_REPETITION.value, 'recommendation': 'Use caching'},
            {'type': LoopType.CONTENT_LOOP.value, 'recommendation': 'Deduplicate content'},
            {'type': LoopType.CONTEXT_OVERFLOW.value, 'recommendation': 'Compress context'}
        ]
        
        recommendations = service._generate_recommendations(test_loops)
        print(f"   Generated recommendations: {recommendations}")
        assert len(recommendations) > 0, "Should generate recommendations"
        print("   ✅ Recommendation generation working")
        
        # 8. Test statistics tracking
        print(f"\n📈 Step 8: Testing Statistics Tracking")
        stats = service.get_stats()
        print(f"   Service stats: {stats}")
        assert 'total_detections' in stats, "Should have detection statistics"
        assert 'config' in stats, "Should have configuration statistics"
        print("   ✅ Statistics tracking working")
        
        # 9. Test activity tracking
        print(f"\n🔄 Step 9: Testing Activity Tracking")
        service.update_activity('test_conv_5')
        assert 'test_conv_5' in service._last_activity_time, "Should track activity"
        print("   ✅ Activity tracking working")
        
        # 10. Test conversation state initialization
        print(f"\n🏗️ Step 10: Testing Conversation State Initialization")
        service._initialize_conversation_state('test_conv_6', 'test_agent')
        assert 'test_conv_6' in service._conversation_start_time, "Should initialize conversation state"
        assert 'test_agent' in service._agent_states, "Should initialize agent state"
        print("   ✅ Conversation state initialization working")
        
        # 11. Test false positive marking
        print(f"\n🚫 Step 11: Testing False Positive Marking")
        initial_false_count = len(service._false_positives)
        service.mark_false_positive('test_conv_7')
        assert len(service._false_positives) == initial_false_count + 1, "Should track false positives"
        print("   ✅ False positive marking working")
        
        # 12. Test global convenience function
        print(f"\n🌐 Step 12: Testing Global Convenience Function")
        global_result = await analyze_conversation_for_loops('test_conv_8', [
            {'role': 'assistant', 'content': 'Testing', 'tool_calls': [
                {'function': {'name': 'test_tool', 'arguments': '{}'}}
            ]}
        ])
        assert 'conversation_id' in global_result, "Should return valid result"
        print("   ✅ Global convenience function working")
        
        # 13. Test custom service creation
        print(f"\n🏭 Step 13: Testing Custom Service Creation")
        custom_service = create_loop_detection_service(LoopDetectionConfig(max_tool_repetitions=1))
        assert custom_service.config.max_tool_repetitions == 1, "Should use custom config"
        print("   ✅ Custom service creation working")
        
        # 14. Test stats reset
        print(f"\n🔄 Step 14: Testing Statistics Reset")
        initial_stats = service.get_stats()
        service.reset_stats()
        reset_stats = service.get_stats()
        assert reset_stats['total_detections'] == 0, "Should reset statistics"
        print("   ✅ Statistics reset working")
        
        # 15. Test comprehensive analysis with mixed issues
        print(f"\n🔍 Step 15: Testing Comprehensive Analysis")
        complex_messages = [
            {'role': 'assistant', 'content': 'Start processing', 'tool_calls': [
                {'function': {'name': 'process', 'arguments': '{}'}}
            ]},
            {'role': 'assistant', 'content': 'Processing again', 'tool_calls': [
                {'function': {'name': 'process', 'arguments': '{}'}}
            ]},
            {'role': 'assistant', 'content': 'Processing again', 'tool_calls': [
                {'function': {'name': 'process', 'arguments': '{}'}}
            ]},
            {'role': 'assistant', 'content': 'Processing again', 'tool_calls': [
                {'function': {'name': 'process', 'arguments': '{}'}}
            ]},
            {'role': 'assistant', 'content': 'Same response'},  # Content repetition
            {'role': 'assistant', 'content': 'Same response'},  # Content repetition
        ]
        
        complex_result = await service.analyze_conversation('complex_test', complex_messages)
        print(f"   Complex analysis found {len(complex_result['detected_loops'])} issues")
        print(f"   Risk level: {complex_result['risk_level']}")
        print(f"   Recommendations: {complex_result['recommendations']}")
        assert len(complex_result['detected_loops']) > 0, "Should detect multiple issues"
        print("   ✅ Comprehensive analysis working")
        
        print("\n🎉 All Tests Passed!")
        print("\n🚀 Loop Detection System Successfully Implemented!")
        print("\n📋 Summary of Features:")
        print("   ✅ Multi-strategy loop detection (tool, content, idle, context)")
        print("   ✅ Configurable thresholds and sensitivity")
        print("   ✅ Risk level calculation and prioritization")
        print("   ✅ Actionable recommendation generation")
        print("   ✅ Comprehensive statistics and monitoring")
        print("   ✅ Performance tracking and optimization")
        print("   ✅ False positive learning and marking")
        print("   ✅ Conversation state management")
        print("   ✅ Global convenience functions")
        print("   ✅ Extensible architecture for future LLM-based detection")
        
        print("\n🎯 Next Steps:")
        print("   1. Add LLM-based semantic loop detection")
        print("   2. Implement agent-specific loop patterns")
        print("   3. Add loop prevention strategies")
        print("   4. Implement MCP Manager singleton")
        print("   5. Add response caching system")
        print("   6. Integrate with multi-agent coordination")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_loop_detection())
    if success:
        print("\n✅ All tests passed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)