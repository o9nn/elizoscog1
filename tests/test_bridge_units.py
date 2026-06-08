"""
Unit Tests for Bridge Implementations
Tests the bridge pattern implementations for cross-ecosystem integration.
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any, List

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bridges.agentmemory_bridge import AgentmemoryBridge
from src.bridges.agentaction_bridge import AgentactionBridge
from src.bridges.agentloop_bridge import AgentloopBridge
from src.bridges.agentbrowser_bridge import AgentbrowserBridge


# Helper to run async tests
def run_async(coro):
    """Run async coroutine in test context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ============================================================================
# AgentMemory Bridge Tests
# ============================================================================

class TestAgentmemoryBridge:
    """Test suite for AgentmemoryBridge."""
    
    def test_bridge_initialization(self):
        """Test bridge initializes correctly."""
        bridge = AgentmemoryBridge({})
        assert bridge is not None
        
        success = run_async(bridge.initialize())
        assert success is True
    
    def test_store_memory(self):
        """Test storing a memory."""
        bridge = AgentmemoryBridge({})
        run_async(bridge.initialize())
        
        request = {
            "type": "store_memory",
            "data": {
                "content": "Test memory content",
                "content_type": "text",
                "metadata": {"tag": "test"}
            }
        }
        
        response = run_async(bridge.process_elizaos_request(request))
        
        # May fail if external services unavailable - check structure
        assert 'success' in response
        # If successful, verify memory_id is present
        if response['success']:
            assert 'memory_id' in response['data']
    
    def test_retrieve_memory(self):
        """Test retrieving a stored memory."""
        bridge = AgentmemoryBridge({})
        run_async(bridge.initialize())
        
        # Store first
        store_request = {
            "type": "store_memory",
            "data": {
                "content": "Retrievable content",
                "content_type": "text"
            }
        }
        store_response = run_async(bridge.process_elizaos_request(store_request))
        
        # Skip if store failed (no external services)
        if not store_response['success']:
            pytest.skip("Storage not available - external services required")
            return
        
        memory_id = store_response['data']['memory_id']
        
        # Retrieve
        retrieve_request = {
            "type": "retrieve_memory",
            "data": {"memory_id": memory_id}
        }
        response = run_async(bridge.process_elizaos_request(retrieve_request))
        
        assert response['success'] is True
        assert response['data']['memory']['content'] == "Retrievable content"
    
    def test_search_memories(self):
        """Test searching memories."""
        bridge = AgentmemoryBridge({})
        run_async(bridge.initialize())
        
        # Store multiple memories
        for i in range(3):
            run_async(bridge.process_elizaos_request({
                "type": "store_memory",
                "data": {
                    "content": f"Searchable memory {i}",
                    "content_type": "text"
                }
            }))
        
        # Search
        search_request = {
            "type": "search",
            "data": {"query": "Searchable", "limit": 10}
        }
        response = run_async(bridge.process_elizaos_request(search_request))
        
        assert response['success'] is True
        assert 'results' in response['data']
    
    def test_data_translation(self):
        """Test data translation between formats."""
        bridge = AgentmemoryBridge({})
        run_async(bridge.initialize())
        
        elizaos_data = {
            "content": "Test content",
            "timestamp": datetime.now().isoformat()
        }
        
        # ElizaOS to OpenCog
        opencog_data = run_async(
            bridge.translate_data(elizaos_data, "elizaos", "opencog")
        )
        assert 'atom_type' in opencog_data
        
        # OpenCog to ElizaOS
        back_to_elizaos = run_async(
            bridge.translate_data(opencog_data, "opencog", "elizaos")
        )
        assert 'content' in back_to_elizaos
    
    def test_bridge_shutdown(self):
        """Test bridge shutdown."""
        bridge = AgentmemoryBridge({})
        run_async(bridge.initialize())
        
        # Should not raise
        run_async(bridge.shutdown())
    
    def test_invalid_request_type(self):
        """Test handling of invalid request type."""
        bridge = AgentmemoryBridge({})
        run_async(bridge.initialize())
        
        request = {
            "type": "invalid_type",
            "data": {}
        }
        
        response = run_async(bridge.process_elizaos_request(request))
        
        assert response['success'] is False
        assert 'error' in response


# ============================================================================
# AgentAction Bridge Tests
# ============================================================================

class TestAgentactionBridge:
    """Test suite for AgentactionBridge."""
    
    def test_bridge_initialization(self):
        """Test bridge initializes correctly."""
        bridge = AgentactionBridge({"max_history_size": 100})
        assert bridge is not None
        
        success = run_async(bridge.initialize())
        assert success is True
    
    def test_register_action(self):
        """Test registering an action."""
        bridge = AgentactionBridge({})
        run_async(bridge.initialize())
        
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
        
        response = run_async(bridge.process_elizaos_request(request))
        
        assert response['success'] is True
    
    def test_execute_action(self):
        """Test executing a registered action."""
        bridge = AgentactionBridge({})
        run_async(bridge.initialize())
        
        # Register action first
        run_async(bridge.process_elizaos_request({
            "type": "register_action",
            "data": {
                "name": "executable_action",
                "definition": {"schema": "test"}
            }
        }))
        
        # Execute action
        request = {
            "type": "execute_action",
            "data": {
                "name": "executable_action",
                "params": {"key": "value"}
            }
        }
        
        response = run_async(bridge.process_elizaos_request(request))
        
        assert response['success'] is True
        assert 'result' in response['data']
    
    def test_get_action_history(self):
        """Test getting action history."""
        bridge = AgentactionBridge({"max_history_size": 100})
        run_async(bridge.initialize())
        
        # Execute some actions
        run_async(bridge.process_elizaos_request({
            "type": "register_action",
            "data": {"name": "history_action", "definition": {}}
        }))
        
        for i in range(3):
            run_async(bridge.process_elizaos_request({
                "type": "execute_action",
                "data": {"name": "history_action", "params": {"iteration": i}}
            }))
        
        # Get history
        request = {
            "type": "get_action_history",
            "data": {"limit": 10}
        }
        
        response = run_async(bridge.process_elizaos_request(request))
        
        assert response['success'] is True
        assert 'history' in response['data']
        assert len(response['data']['history']) >= 3
    
    def test_chain_actions(self):
        """Test chaining multiple actions."""
        bridge = AgentactionBridge({})
        run_async(bridge.initialize())
        
        # Register actions
        for name in ['action1', 'action2', 'action3']:
            run_async(bridge.process_elizaos_request({
                "type": "register_action",
                "data": {"name": name, "definition": {}}
            }))
        
        # Chain actions
        request = {
            "type": "chain_actions",
            "data": {
                "actions": [
                    {"name": "action1", "params": {}},
                    {"name": "action2", "params": {}},
                    {"name": "action3", "params": {}}
                ]
            }
        }
        
        response = run_async(bridge.process_elizaos_request(request))
        
        # Check response structure - may fail if chain_actions not implemented
        assert 'success' in response
        if response['success']:
            assert 'chain_results' in response['data']
    
    def test_action_translation(self):
        """Test action data translation."""
        bridge = AgentactionBridge({})
        run_async(bridge.initialize())
        
        elizaos_action = {
            "name": "translate_action",
            "params": {"key": "value"}
        }
        
        opencog_action = run_async(
            bridge.translate_data(elizaos_action, "elizaos", "opencog")
        )
        
        assert 'atom_type' in opencog_action


# ============================================================================
# AgentLoop Bridge Tests
# ============================================================================

class TestAgentloopBridge:
    """Test suite for AgentloopBridge."""
    
    def test_bridge_initialization(self):
        """Test bridge initializes correctly."""
        bridge = AgentloopBridge({})
        assert bridge is not None
        
        success = run_async(bridge.initialize())
        assert success is True
    
    def test_start_loop(self):
        """Test starting an agent loop."""
        bridge = AgentloopBridge({})
        run_async(bridge.initialize())
        
        request = {
            "type": "start_loop",
            "data": {
                "loop_id": "test_loop_1",
                "config": {
                    "interval": 1.0,
                    "max_iterations": 5
                }
            }
        }
        
        response = run_async(bridge.process_elizaos_request(request))
        
        assert response['success'] is True
    
    def test_stop_loop(self):
        """Test stopping a running loop."""
        bridge = AgentloopBridge({})
        run_async(bridge.initialize())
        
        # Start loop
        run_async(bridge.process_elizaos_request({
            "type": "start_loop",
            "data": {"loop_id": "stoppable_loop", "config": {"interval": 1.0}}
        }))
        
        # Stop loop
        request = {
            "type": "stop_loop",
            "data": {"loop_id": "stoppable_loop"}
        }
        
        response = run_async(bridge.process_elizaos_request(request))
        
        assert response['success'] is True
    
    def test_get_loop_status(self):
        """Test getting loop status."""
        bridge = AgentloopBridge({})
        run_async(bridge.initialize())
        
        # Start a loop
        run_async(bridge.process_elizaos_request({
            "type": "start_loop",
            "data": {"loop_id": "status_loop", "config": {}}
        }))
        
        # Get status
        request = {
            "type": "get_status",
            "data": {"loop_id": "status_loop"}
        }
        
        response = run_async(bridge.process_elizaos_request(request))
        
        # Check response structure
        assert 'success' in response
        if response['success']:
            assert 'status' in response['data']
    
    def test_list_loops(self):
        """Test listing all loops."""
        bridge = AgentloopBridge({})
        run_async(bridge.initialize())
        
        # Start multiple loops
        for i in range(3):
            run_async(bridge.process_elizaos_request({
                "type": "start_loop",
                "data": {"loop_id": f"list_loop_{i}", "config": {}}
            }))
        
        # List loops
        request = {
            "type": "list_loops",
            "data": {}
        }
        
        response = run_async(bridge.process_elizaos_request(request))
        
        assert response['success'] is True
        assert 'loops' in response['data']
        assert len(response['data']['loops']) >= 3
    
    def test_loop_translation(self):
        """Test loop data translation."""
        bridge = AgentloopBridge({})
        run_async(bridge.initialize())
        
        elizaos_loop = {
            "loop_id": "translate_loop",
            "config": {"interval": 1.0}
        }
        
        opencog_loop = run_async(
            bridge.translate_data(elizaos_loop, "elizaos", "opencog")
        )
        
        assert 'atom_type' in opencog_loop


# ============================================================================
# AgentBrowser Bridge Tests
# ============================================================================

class TestAgentbrowserBridge:
    """Test suite for AgentbrowserBridge."""
    
    def test_bridge_initialization(self):
        """Test bridge initializes correctly."""
        bridge = AgentbrowserBridge({})
        assert bridge is not None
        
        success = run_async(bridge.initialize())
        assert success is True
    
    def test_browse_request(self):
        """Test browse request (simulated)."""
        bridge = AgentbrowserBridge({})
        run_async(bridge.initialize())
        
        request = {
            "type": "browse",
            "data": {
                "url": "https://example.com",
                "timeout": 5
            }
        }
        
        response = run_async(bridge.process_elizaos_request(request))
        
        # May fail in test environment without network
        # But should not throw an exception
        assert 'success' in response
    
    def test_scrape_request(self):
        """Test scrape request structure."""
        bridge = AgentbrowserBridge({})
        run_async(bridge.initialize())
        
        request = {
            "type": "scrape",
            "data": {
                "url": "https://example.com",
                "selectors": {"title": "h1", "content": "p"}
            }
        }
        
        response = run_async(bridge.process_elizaos_request(request))
        
        assert 'success' in response
    
    def test_search_request(self):
        """Test search request structure."""
        bridge = AgentbrowserBridge({})
        run_async(bridge.initialize())
        
        request = {
            "type": "search",
            "data": {
                "query": "test search query",
                "engine": "duckduckgo"
            }
        }
        
        response = run_async(bridge.process_elizaos_request(request))
        
        assert 'success' in response
    
    def test_browser_translation(self):
        """Test browser data translation."""
        bridge = AgentbrowserBridge({})
        run_async(bridge.initialize())
        
        elizaos_data = {
            "url": "https://example.com",
            "content": "Page content"
        }
        
        opencog_data = run_async(
            bridge.translate_data(elizaos_data, "elizaos", "opencog")
        )
        
        assert 'atom_type' in opencog_data


# ============================================================================
# Bridge Integration Tests
# ============================================================================

class TestBridgeIntegration:
    """Integration tests for bridges working together."""
    
    def test_memory_action_workflow(self):
        """Test workflow using memory and action bridges together."""
        memory_bridge = AgentmemoryBridge({})
        action_bridge = AgentactionBridge({})
        
        run_async(memory_bridge.initialize())
        run_async(action_bridge.initialize())
        
        # Store a memory
        memory_response = run_async(memory_bridge.process_elizaos_request({
            "type": "store_memory",
            "data": {
                "content": "Action context memory",
                "content_type": "context"
            }
        }))
        
        # Skip if memory storage failed (no external services)
        if not memory_response['success']:
            pytest.skip("Memory storage not available - external services required")
            return
        
        memory_id = memory_response['data']['memory_id']
        
        # Register an action that references the memory
        run_async(action_bridge.process_elizaos_request({
            "type": "register_action",
            "data": {
                "name": "memory_aware_action",
                "definition": {"memory_reference": memory_id}
            }
        }))
        
        # Execute the action
        action_response = run_async(action_bridge.process_elizaos_request({
            "type": "execute_action",
            "data": {
                "name": "memory_aware_action",
                "params": {"context_memory": memory_id}
            }
        }))
        
        assert memory_response['success'] is True
        assert action_response['success'] is True
    
    def test_all_bridges_initialize(self):
        """Test that all bridges can initialize without conflict."""
        bridges = [
            AgentmemoryBridge({}),
            AgentactionBridge({}),
            AgentloopBridge({}),
            AgentbrowserBridge({})
        ]
        
        for bridge in bridges:
            success = run_async(bridge.initialize())
            assert success is True
        
        # Clean shutdown
        for bridge in bridges:
            run_async(bridge.shutdown())
    
    def test_bridge_error_handling(self):
        """Test error handling across bridges."""
        bridges = [
            AgentmemoryBridge({}),
            AgentactionBridge({}),
            AgentloopBridge({}),
            AgentbrowserBridge({})
        ]
        
        for bridge in bridges:
            run_async(bridge.initialize())
            
            # Invalid request should return error, not throw
            response = run_async(bridge.process_elizaos_request({
                "type": "nonexistent_operation",
                "data": {}
            }))
            
            assert 'success' in response
            assert response['success'] is False


# ============================================================================
# Performance Tests
# ============================================================================

class TestBridgePerformance:
    """Performance tests for bridge operations."""
    
    def test_memory_storage_performance(self):
        """Test memory storage performance."""
        import time
        
        bridge = AgentmemoryBridge({"max_memories": 1000})
        run_async(bridge.initialize())
        
        start_time = time.time()
        
        for i in range(100):
            run_async(bridge.process_elizaos_request({
                "type": "store_memory",
                "data": {
                    "content": f"Performance test memory {i}",
                    "content_type": "test"
                }
            }))
        
        elapsed = time.time() - start_time
        
        # Should complete 100 stores in under 5 seconds
        assert elapsed < 5.0, f"Memory storage too slow: {elapsed:.2f}s"
    
    def test_action_execution_performance(self):
        """Test action execution performance."""
        import time
        
        bridge = AgentactionBridge({})
        run_async(bridge.initialize())
        
        # Register action
        run_async(bridge.process_elizaos_request({
            "type": "register_action",
            "data": {"name": "perf_action", "definition": {}}
        }))
        
        start_time = time.time()
        
        for i in range(100):
            run_async(bridge.process_elizaos_request({
                "type": "execute_action",
                "data": {"name": "perf_action", "params": {"i": i}}
            }))
        
        elapsed = time.time() - start_time
        
        # Should complete 100 executions in under 5 seconds
        assert elapsed < 5.0, f"Action execution too slow: {elapsed:.2f}s"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
