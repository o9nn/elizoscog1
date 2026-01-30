#!/usr/bin/env python3
"""
Test script for agent bridges: agentaction, agentloop, and agentbrowser
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.bridges.agentaction_bridge import AgentactionBridge
from src.bridges.agentloop_bridge import AgentloopBridge
from src.bridges.agentbrowser_bridge import AgentbrowserBridge


async def test_agentaction_bridge():
    """Test agentaction bridge implementation"""
    print("\n=== Testing AgentAction Bridge ===")
    
    bridge = AgentactionBridge({"max_history_size": 100})
    
    # Test initialization
    success = await bridge.initialize()
    print(f"✓ Initialization: {'SUCCESS' if success else 'FAILED'}")
    assert success, "Bridge initialization failed"
    
    # Test register action
    request = {
        "type": "register_action",
        "data": {
            "name": "test_action",
            "definition": {
                "schema": "test_schema",
                "params": ["param1", "param2"]
            }
        }
    }
    response = await bridge.process_elizaos_request(request)
    print(f"✓ Register action: {response['success']}")
    assert response['success'], "Failed to register action"
    
    # Test execute action
    request = {
        "type": "execute_action",
        "data": {
            "name": "test_action",
            "params": {"param1": "value1"}
        }
    }
    response = await bridge.process_elizaos_request(request)
    print(f"✓ Execute action: {response['success']}")
    assert response['success'], "Failed to execute action"
    
    # Test get action history
    request = {
        "type": "get_action_history",
        "data": {"limit": 5}
    }
    response = await bridge.process_elizaos_request(request)
    print(f"✓ Get history: {response['success']}, {len(response['data']['history'])} entries")
    assert response['success'], "Failed to get action history"
    
    # Test translation
    test_data = {"name": "test", "params": {}}
    translated = await bridge.translate_data(test_data, "elizaos", "opencog")
    print(f"✓ Translation ElizaOS->OpenCog: {translated['atom_type']}")
    
    await bridge.shutdown()
    print("✓ Shutdown successful")


async def test_agentloop_bridge():
    """Test agentloop bridge implementation"""
    print("\n=== Testing AgentLoop Bridge ===")
    
    bridge = AgentloopBridge({"max_history_size": 100})
    
    # Test initialization
    success = await bridge.initialize()
    print(f"✓ Initialization: {'SUCCESS' if success else 'FAILED'}")
    assert success, "Bridge initialization failed"
    
    # Test start loop
    request = {
        "type": "start_loop",
        "data": {
            "loop_id": "test_loop_1",
            "config": {
                "interval": 1.0,
                "max_iterations": 10
            }
        }
    }
    response = await bridge.process_elizaos_request(request)
    print(f"✓ Start loop: {response['success']}")
    assert response['success'], "Failed to start loop"
    
    # Test list loops
    request = {
        "type": "list_loops",
        "data": {}
    }
    response = await bridge.process_elizaos_request(request)
    print(f"✓ List loops: {response['success']}, {len(response['data']['loops'])} active")
    assert response['success'], "Failed to list loops"
    assert len(response['data']['loops']) > 0, "No active loops found"
    
    # Test get loop status
    request = {
        "type": "get_loop_status",
        "data": {"loop_id": "test_loop_1"}
    }
    response = await bridge.process_elizaos_request(request)
    print(f"✓ Get loop status: {response['success']}, status={response['data']['status']['status']}")
    assert response['success'], "Failed to get loop status"
    
    # Test stop loop
    request = {
        "type": "stop_loop",
        "data": {"loop_id": "test_loop_1"}
    }
    response = await bridge.process_elizaos_request(request)
    print(f"✓ Stop loop: {response['success']}")
    assert response['success'], "Failed to stop loop"
    
    # Test translation
    test_data = {"loop_id": "test", "status": "running"}
    translated = await bridge.translate_data(test_data, "elizaos", "opencog")
    print(f"✓ Translation ElizaOS->OpenCog: {translated['atom_type']}")
    
    await bridge.shutdown()
    print("✓ Shutdown successful")


async def test_agentbrowser_bridge():
    """Test agentbrowser bridge implementation"""
    print("\n=== Testing AgentBrowser Bridge ===")
    
    bridge = AgentbrowserBridge({"max_cache_size": 50})
    
    # Test initialization
    success = await bridge.initialize()
    print(f"✓ Initialization: {'SUCCESS' if success else 'FAILED'}")
    assert success, "Bridge initialization failed"
    
    # Test browse URL
    request = {
        "type": "browse_url",
        "data": {
            "url": "https://example.com",
            "session_id": "test_session"
        }
    }
    response = await bridge.process_elizaos_request(request)
    print(f"✓ Browse URL: {response['success']}")
    assert response['success'], "Failed to browse URL"
    
    # Test scrape page
    request = {
        "type": "scrape_page",
        "data": {
            "url": "https://example.com/data",
            "selector": ".content"
        }
    }
    response = await bridge.process_elizaos_request(request)
    print(f"✓ Scrape page: {response['success']}")
    assert response['success'], "Failed to scrape page"
    
    # Test extract data
    request = {
        "type": "extract_data",
        "data": {
            "url": "https://example.com/info",
            "rules": {
                "title": {"selector": "h1"},
                "price": {"selector": ".price"}
            }
        }
    }
    response = await bridge.process_elizaos_request(request)
    print(f"✓ Extract data: {response['success']}, {len(response['data']['extracted']['data'])} fields")
    assert response['success'], "Failed to extract data"
    
    # Test OpenCog atomization
    request = {
        "type": "browse_and_atomize",
        "data": {
            "url": "https://example.com/knowledge"
        }
    }
    response = await bridge.process_opencog_request(request)
    print(f"✓ Browse and atomize: {response['success']}, {len(response['data']['atoms'])} atoms")
    assert response['success'], "Failed to browse and atomize"
    
    # Test translation
    test_data = {"url": "https://example.com", "content": "test content"}
    translated = await bridge.translate_data(test_data, "elizaos", "opencog")
    print(f"✓ Translation ElizaOS->OpenCog: {translated['atom_type']}")
    
    await bridge.shutdown()
    print("✓ Shutdown successful")


async def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Agent Bridges Implementation")
    print("=" * 60)
    
    try:
        await test_agentaction_bridge()
        await test_agentloop_bridge()
        await test_agentbrowser_bridge()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
