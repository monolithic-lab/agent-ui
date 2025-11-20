#!/usr/bin/env python3
# test_agent_registry.py
"""
Comprehensive test suite for the Agent Registry system
Tests the enterprise-grade agent registry pattern matching Qwen-Agent
"""

import sys
import os
import asyncio
import traceback
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_agent_registry():
    """Test the new agent registry system comprehensively"""
    
    print("🤖 Testing Agent-UI Agent Registry System")
    print("=" * 50)
    
    # Import the registry functions
    from agents import (
        AGENT_REGISTRY, 
        get_agent_registry, 
        list_available_agents,
        create_agent,
        get_agent_instance,
        get_agent_info,
        reload_agent_registry,
        AgentConfig,
        BaseAgent
    )
    
    try:
        # 1. Check registry population
        print(f"\n📋 Step 1: Checking Registry Population")
        print(f"   Registered agents: {list_available_agents()}")
        assert len(AGENT_REGISTRY) > 0, "Registry should have at least one agent"
        expected_agents = ['fncall_agent', 'assistant']
        for agent in expected_agents:
            assert agent in AGENT_REGISTRY, f"{agent} should be registered"
        print("   ✅ Registry properly populated")
        
        # 2. Test agent creation via factory
        print(f"\n🔧 Step 2: Testing Agent Factory Pattern")
        config = AgentConfig(
            name="test_agent",
            description="Test agent",
            system_message="You are a test agent"
        )
        fncall_agent = create_agent('fncall_agent', config=config)
        print(f"   ✅ Created agent via factory: {fncall_agent.config.name}")
        assert fncall_agent.config.name == "test_agent", "Agent name should match"
        
        # 3. Test singleton pattern
        print(f"\n🏠 Step 3: Testing Singleton Pattern")
        instance1 = get_agent_instance('assistant', config=config)
        instance2 = get_agent_instance('assistant', config=config)
        assert instance1 is instance2, "Should be same instance (singleton)"
        print("   ✅ Singleton working correctly")
        
        # 4. Test agent information
        print(f"\n📊 Step 4: Testing Agent Information")
        for agent_name in list_available_agents():
            info = get_agent_info(agent_name)
            print(f"   Agent info for {agent_name}:")
            for key, value in info.items():
                print(f"     {key}: {value}")
        print("   ✅ Agent information retrieved")
        
        # 5. Test agent creation from registry
        print(f"\n🔍 Step 5: Testing Agent Creation from Registry")
        assistant_agent = BaseAgent.from_registry('assistant', config=config)
        assert isinstance(assistant_agent, BaseAgent), "Should be BaseAgent instance"
        print("   ✅ Agent creation from registry working")
        
        # 6. Test agent info methods
        print(f"\n🔍 Step 6: Testing Agent Info Methods")
        agent_info = fncall_agent.get_info()
        print(f"   Agent info: {agent_info}")
        assert 'name' in agent_info and 'agent_type' in agent_info, "Should have basic info"
        
        stats = fncall_agent.execution_stats
        print(f"   Execution stats: {stats}")
        assert 'name' in stats and 'agent_type' in stats, "Should have execution stats"
        print("   ✅ Agent info methods working")
        
        # 7. Test error handling
        print(f"\n❌ Step 7: Testing Error Handling")
        try:
            create_agent('non_existent_agent', config=config)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            print(f"   ✅ Correctly caught error: {str(e)}")
        
        # 8. Test registry reload
        print(f"\n🔄 Step 8: Testing Registry Reload")
        initial_agents = list_available_agents()
        reload_agent_registry()
        reloaded_agents = list_available_agents()
        assert initial_agents == reloaded_agents, "Agents should be same after reload"
        print("   ✅ Registry reload working")
        
        # 9. Test agent hierarchy
        print(f"\n👥 Step 9: Testing Agent Hierarchy")
        assert isinstance(fncall_agent, BaseAgent), "FnCallAgent should inherit from BaseAgent"
        assert isinstance(assistant_agent, BaseAgent), "Assistant should inherit from BaseAgent"
        print("   ✅ Agent hierarchy working")
        
        # 10. Test agent registry copy
        print(f"\n📋 Step 10: Testing Registry Copy")
        registry_copy = get_agent_registry()
        assert isinstance(registry_copy, dict), "Should return dictionary"
        assert len(registry_copy) > 0, "Copy should not be empty"
        print("   ✅ Registry copy working")
        
        # 11. Test agent metadata
        print(f"\n📝 Step 11: Testing Agent Metadata")
        for agent_name in list_available_agents():
            agent_class = AGENT_REGISTRY[agent_name]
            metadata = getattr(agent_class, '_registry_name', None)
            assert metadata is not None, f"Agent {agent_name} should have registry metadata"
            print(f"   ✅ {agent_name} has metadata: {metadata}")
        
        print("\n🎉 All Tests Passed!")
        print("\n🚀 Agent Registry System Successfully Implemented!")
        print("\n📋 Summary of Features:")
        print("   ✅ Dynamic agent registration with __agent_name__")
        print("   ✅ Factory pattern for agent creation")
        print("   ✅ Singleton pattern for performance")
        print("   ✅ Agent information and metadata system")
        print("   ✅ Comprehensive error handling")
        print("   ✅ Agent hierarchy (BaseAgent → FnCallAgent → Assistant)")
        print("   ✅ Registry reload capability")
        print("   ✅ Agent statistics and monitoring")
        
        print("\n🎯 Next Steps:")
        print("   1. Add multi-agent coordination system")
        print("   2. Implement group chat functionality")
        print("   3. Add agent communication protocols")
        print("   4. Implement enhanced loop detection")
        print("   5. Add MCP Manager singleton")
        print("   6. Implement response caching")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = test_agent_registry()
    if success:
        print("\n✅ All tests passed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)