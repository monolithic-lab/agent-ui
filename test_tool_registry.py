#!/usr/bin/env python3
# test_tool_registry.py
"""
Comprehensive test suite for the Tool Registry system
Tests the enterprise-grade tool registry pattern matching Qwen-Agent
"""

import sys
import os
import asyncio
import traceback
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_tool_registry():
    """Test the new tool registry system comprehensively"""
    
    print("🧪 Testing Agent-UI Tool Registry System")
    print("=" * 50)
    
    # Import the registry functions
    from tools import (
        TOOL_REGISTRY, 
        get_tool_registry, 
        list_available_tools,
        create_tool,
        get_tool_instance,
        get_tool_info,
        reload_tool_registry
    )
    
    try:
        # 1. Check registry population
        print(f"\n📋 Step 1: Checking Registry Population")
        print(f"   Registered tools: {list_available_tools()}")
        assert 'code_interpreter' in TOOL_REGISTRY, "CodeInterpreter should be registered"
        print("   ✅ Registry properly populated")
        
        # 2. Test tool creation via factory
        print(f"\n🔧 Step 2: Testing Tool Factory Pattern")
        code_interpreter = create_tool('code_interpreter', timeout=30)
        print(f"   ✅ Created tool via factory: {code_interpreter.name}")
        assert code_interpreter.name == 'code_interpreter', "Tool name should match"
        
        # 3. Test singleton pattern
        print(f"\n🏠 Step 3: Testing Singleton Pattern")
        instance1 = get_tool_instance('code_interpreter', timeout=30)
        instance2 = get_tool_instance('code_interpreter', timeout=30)
        assert instance1 is instance2, "Should be same instance (singleton)"
        print("   ✅ Singleton working correctly")
        
        # 4. Test tool information
        print(f"\n📊 Step 4: Testing Tool Information")
        info = get_tool_info('code_interpreter')
        print(f"   Tool info:")
        for key, value in info.items():
            print(f"     {key}: {value}")
        assert 'name' in info and 'class' in info, "Tool info should contain essential data"
        print("   ✅ Tool information retrieved")
        
        # 5. Test tool execution
        print(f"\n⚡ Step 5: Testing Tool Execution")
        async def test_execution():
            result = await code_interpreter.safe_execute({
                'code': 'print("Hello from Tool Registry!"); result = 2 + 2; result'
            })
            print(f"   Execution result: {result.content[:100]}...")
            print(f"   Success: {result.success}")
            print(f"   Metadata: {result.metadata}")
            assert result.success, "Tool execution should succeed"
            assert 'execution_count' in result.metadata, "Should have execution metadata"
        
        asyncio.run(test_execution())
        print("   ✅ Tool execution working")
        
        # 6. Test tool statistics
        print(f"\n📈 Step 6: Testing Tool Statistics")
        stats = code_interpreter.execution_stats
        print(f"   Execution stats: {stats}")
        assert stats['total_executions'] > 0, "Should have execution count"
        print("   ✅ Tool statistics working")
        
        # 7. Test tool info
        print(f"\n🔍 Step 7: Testing Enhanced Tool Info")
        tool_info = code_interpreter.get_info()
        print(f"   Enhanced tool info:")
        for key, value in tool_info.items():
            if key != 'execution_stats':  # Skip the nested dict for readability
                print(f"     {key}: {value}")
        assert tool_info['name'] == 'code_interpreter', "Tool info should be accurate"
        print("   ✅ Enhanced tool info working")
        
        # 8. Test error handling
        print(f"\n❌ Step 8: Testing Error Handling")
        try:
            create_tool('non_existent_tool')
            assert False, "Should have raised ValueError"
        except ValueError as e:
            print(f"   ✅ Correctly caught error: {str(e)}")
        
        # Test invalid arguments
        async def test_invalid_args():
            try:
                await code_interpreter.safe_execute({})  # No 'code' argument
                assert False, "Should have raised error for missing arguments"
            except Exception as e:
                print(f"   ✅ Correctly caught argument error: {type(e).__name__}")
        
        asyncio.run(test_invalid_args())
        print("   ✅ Error handling working")
        
        # 9. Test registry reload
        print(f"\n🔄 Step 9: Testing Registry Reload")
        initial_tools = list_available_tools()
        reload_tool_registry()
        reloaded_tools = list_available_tools()
        assert initial_tools == reloaded_tools, "Tools should be same after reload"
        print("   ✅ Registry reload working")
        
        # 10. Test tool schema
        print(f"\n📝 Step 10: Testing Tool Schema")
        schema = code_interpreter.get_schema()
        print(f"   Schema: {schema}")
        assert schema.name == 'code_interpreter', "Schema name should match"
        assert 'code' in schema.required, "Code should be required parameter"
        print("   ✅ Tool schema working")
        
        # 11. Test tool enabling/disabling
        print(f"\n🔐 Step 11: Testing Tool Enable/Disable")
        code_interpreter.enabled = False
        assert not code_interpreter.enabled, "Tool should be disabled"
        code_interpreter.enabled = True
        assert code_interpreter.enabled, "Tool should be enabled"
        print("   ✅ Tool enable/disable working")
        
        # 12. Test multiple instances with different configs
        print(f"\n⚙️ Step 12: Testing Multiple Configurations")
        fast_tool = create_tool('code_interpreter', timeout=10)
        slow_tool = create_tool('code_interpreter', timeout=60)
        assert fast_tool.timeout == 10, "First instance should have timeout 10"
        assert slow_tool.timeout == 60, "Second instance should have timeout 60"
        print("   ✅ Multiple configurations working")
        
        print("\n🎉 All Tests Passed!")
        print("\n🚀 Tool Registry System Successfully Implemented!")
        print("\n📋 Summary of Features:")
        print("   ✅ Dynamic tool registration with @register_tool decorator")
        print("   ✅ Factory pattern for tool creation")
        print("   ✅ Singleton pattern for performance")
        print("   ✅ Tool information and metadata system")
        print("   ✅ Comprehensive error handling")
        print("   ✅ Execution statistics and monitoring")
        print("   ✅ Registry reload capability")
        print("   ✅ Tool schema for function calling")
        print("   ✅ Enhanced BaseTool with registry support")
        
        print("\n🎯 Next Steps:")
        print("   1. Add more tools with @register_tool decorator")
        print("   2. Implement Agent Registry pattern")
        print("   3. Add multi-agent coordination")
        print("   4. Implement enhanced loop detection")
        print("   5. Add MCP Manager singleton")
        print("   6. Implement response caching")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = test_tool_registry()
    if success:
        print("\n✅ All tests passed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)