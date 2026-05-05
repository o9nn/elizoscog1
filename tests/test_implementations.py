#!/usr/bin/env python3
"""
Test script to verify ElizaOS bridge implementations
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bridges.agentmemory_bridge import AgentmemoryBridge
from bridges.easycompletion_bridge import EasycompletionBridge 
from bridges.agentbrowser_bridge import AgentbrowserBridge
from bridges.agentloop_bridge import AgentloopBridge
from bridges.agentaction_bridge import AgentactionBridge

async def test_bridge_initialization():
    """Test that all bridges can be initialized"""
    print("🧪 Testing ElizaOS Bridge Implementations...")
    
    # Test configuration
    test_config = {
        'elizaos': {'endpoint': 'http://localhost:3000'},
        'opencog': {'host': 'localhost', 'port': 17001},
        'gnucash': {'file_path': '/tmp/test.gnucash'}
    }
    
    # Test bridges
    bridges = {
        'agentmemory': AgentmemoryBridge(test_config),
        'easycompletion': EasycompletionBridge(test_config),
        'agentbrowser': AgentbrowserBridge(test_config),
        'agentloop': AgentloopBridge(test_config),
        'agentaction': AgentactionBridge(test_config)
    }
    
    # Test initialization
    results = {}
    for name, bridge in bridges.items():
        try:
            success = await bridge.initialize()
            results[name] = success
            status = "✅" if success else "❌"
            print(f"  {status} {name} bridge: {'Initialized' if success else 'Failed'}")
        except Exception as e:
            results[name] = False
            print(f"  ❌ {name} bridge: Failed with error - {e}")
    
    # Test basic operations
    print("\n🧪 Testing Basic Operations...")
    
    # Test agentmemory operations
    if results.get('agentmemory'):
        try:
            memory_bridge = bridges['agentmemory']
            
            # Test ElizaOS request
            elizaos_request = {
                'operation': 'store_memory',
                'agent_id': 'test_agent_1',
                'data': {'content': 'Test memory content', 'type': 'test'}
            }
            
            response = await memory_bridge.process_elizaos_request(elizaos_request)
            if response.get('success'):
                print("  ✅ agentmemory: ElizaOS memory storage working")
            else:
                print("  ❌ agentmemory: ElizaOS memory storage failed")
                
            # Test data translation
            test_data = {'memories': [{'id': '1', 'content': 'test'}]}
            translated = await memory_bridge.translate_data(test_data, 'elizaos', 'opencog')
            if 'atomspace_data' in translated:
                print("  ✅ agentmemory: Data translation working")
            else:
                print("  ❌ agentmemory: Data translation failed")
                
        except Exception as e:
            print(f"  ❌ agentmemory operations failed: {e}")
    
    # Test easycompletion operations  
    if results.get('easycompletion'):
        try:
            completion_bridge = bridges['easycompletion']
            
            # Test ElizaOS request
            completion_request = {
                'operation': 'text_completion',
                'agent_id': 'test_agent_1', 
                'data': {'prompt': 'Hello, how are you?', 'context': 'test'}
            }
            
            response = await completion_bridge.process_elizaos_request(completion_request)
            if response.get('success'):
                print("  ✅ easycompletion: ElizaOS text completion working")
            else:
                print("  ❌ easycompletion: ElizaOS text completion failed")
                
        except Exception as e:
            print(f"  ❌ easycompletion operations failed: {e}")
    
    # Summary
    print(f"\n📊 Test Results:")
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    print(f"  Bridges initialized: {successful}/{total}")
    
    if successful == total:
        print("  🎉 All ElizaOS implementations completed successfully!")
    else:
        print("  ⚠️  Some implementations need attention")
    
    # Cleanup
    for bridge in bridges.values():
        if hasattr(bridge, 'initialized') and bridge.initialized:
            await bridge.shutdown()

if __name__ == "__main__":
    asyncio.run(test_bridge_initialization())