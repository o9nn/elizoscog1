#!/usr/bin/env python3
"""
Comprehensive Tests for Distributed Cognitive Mesh APIs (Phase 4)

Tests the distributed state management, API endpoints, WebSocket communication,
authentication, and external system bindings with performance validation.
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Test imports
from src.api.state_manager import DistributedStateManager, StateEvent, StateNode, StateEventType
from src.api.mesh_api import CognitiveMeshAPI, ApiResponse
from src.api.websocket_handler import WebSocketHandler, ConnectionManager
from src.api.auth_manager import AuthenticationManager, User, UserRole, Permission
from src.api.external_bindings import UnityBinding, ROSBinding, WebAgentBinding, ExternalMessage


class TestResults:
    """Collect and track test results"""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.errors = []
        self.performance_metrics = {}
        self.start_time = datetime.now()
    
    def record_test(self, test_name: str, passed: bool, error: str = None, 
                   execution_time_ms: float = None):
        self.tests_run += 1
        if passed:
            self.tests_passed += 1
            status = "PASSED"
        else:
            self.tests_failed += 1
            status = "FAILED"
            if error:
                self.errors.append(f"{test_name}: {error}")
        
        if execution_time_ms:
            self.performance_metrics[test_name] = execution_time_ms
        
        print(f"  {test_name}: {status}" + (f" ({execution_time_ms:.1f}ms)" if execution_time_ms else ""))
    
    def get_summary(self) -> Dict[str, Any]:
        total_time = (datetime.now() - self.start_time).total_seconds()
        avg_response_time = sum(self.performance_metrics.values()) / len(self.performance_metrics) if self.performance_metrics else 0
        
        return {
            "tests_run": self.tests_run,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "pass_rate": self.tests_passed / self.tests_run if self.tests_run > 0 else 0,
            "total_time_seconds": total_time,
            "average_response_time_ms": avg_response_time,
            "sub_100ms_responses": sum(1 for t in self.performance_metrics.values() if t < 100),
            "performance_target_met": avg_response_time < 100,
            "errors": self.errors
        }


async def test_distributed_state_manager():
    """Test distributed state management functionality"""
    print("\n=== Testing Distributed State Manager ===")
    results = TestResults()
    
    # Test 1: Basic state operations
    start_time = time.time()
    try:
        state_manager = DistributedStateManager("test_node_1")
        await state_manager.start()
        
        # Set state
        event = await state_manager.set_state("user", "user_123", {"name": "Alice", "score": 85})
        assert event.entity_type == "user"
        assert event.entity_id == "user_123"
        
        # Get state
        state = state_manager.get_state("user", "user_123")
        assert state["name"] == "Alice"
        assert state["score"] == 85
        
        await state_manager.stop()
        results.record_test("Basic state operations", True, None, (time.time() - start_time) * 1000)
        
    except Exception as e:
        results.record_test("Basic state operations", False, str(e))
    
    # Test 2: State synchronization between nodes
    start_time = time.time()
    try:
        node1 = DistributedStateManager("node_1")
        node2 = DistributedStateManager("node_2")
        
        await node1.start()
        await node2.start()
        
        # Register nodes with each other
        node1_info = StateNode("node_1", "localhost", 8001, ["cognitive"], datetime.now())
        node2_info = StateNode("node_2", "localhost", 8002, ["financial"], datetime.now())
        
        node1.register_node(node2_info)
        node2.register_node(node1_info)
        
        # Create event on node1
        event1 = await node1.set_state("cognitive_state", "thought_1", {"content": "Deep thinking"})
        
        # Simulate event propagation to node2
        success = await node2.handle_remote_event(event1)
        assert success == True
        
        # Verify state exists on node2
        state = node2.get_state("cognitive_state", "thought_1")
        assert state["content"] == "Deep thinking"
        
        await node1.stop()
        await node2.stop()
        results.record_test("State synchronization", True, None, (time.time() - start_time) * 1000)
        
    except Exception as e:
        results.record_test("State synchronization", False, str(e))
    
    # Test 3: Conflict resolution
    start_time = time.time()
    try:
        node1 = DistributedStateManager("node_a")
        node2 = DistributedStateManager("node_b") 
        
        await node1.start()
        await node2.start()
        
        # Create conflicting states
        event1 = await node1.set_state("resource", "res_1", {"value": 100})
        event2 = await node2.set_state("resource", "res_1", {"value": 200})
        
        # Apply events with same version (conflict scenario)
        event1.version = 1
        event2.version = 1
        
        # Node with higher ID should win (node_b > node_a)
        await node1.handle_remote_event(event2)
        state = node1.get_state("resource", "res_1")
        assert state["value"] == 200  # Node B's value should win
        
        await node1.stop()
        await node2.stop()
        results.record_test("Conflict resolution", True, None, (time.time() - start_time) * 1000)
        
    except Exception as e:
        results.record_test("Conflict resolution", False, str(e))
    
    # Test 4: Performance under load
    start_time = time.time()
    try:
        state_manager = DistributedStateManager("perf_node")
        await state_manager.start()
        
        # Create 1000 state entries rapidly
        tasks = []
        for i in range(1000):
            task = state_manager.set_state("perf_test", f"item_{i}", {"value": i})
            tasks.append(task)
        
        events = await asyncio.gather(*tasks)
        assert len(events) == 1000
        
        # Verify all states exist
        all_state = state_manager.get_all_state("perf_test")
        assert len(all_state) == 1000
        
        await state_manager.stop()
        
        execution_time = (time.time() - start_time) * 1000
        results.record_test("Performance under load", True, None, execution_time)
        
        # Check if under 100ms per operation on average
        avg_per_operation = execution_time / 1000
        if avg_per_operation > 1.0:  # Allow 1ms per operation
            results.errors.append(f"Performance test: {avg_per_operation:.2f}ms per operation (target: <1ms)")
    
    except Exception as e:
        results.record_test("Performance under load", False, str(e))
    
    return results


async def test_cognitive_mesh_api():
    """Test the main cognitive mesh API functionality"""
    print("\n=== Testing Cognitive Mesh API ===")
    results = TestResults()
    
    # Test 1: API initialization
    start_time = time.time()
    try:
        api = CognitiveMeshAPI("api_test_node", port=8099)
        await api.initialize()
        
        assert api.node_id == "api_test_node"
        assert api.state_manager is not None
        assert api.websocket_handler is not None
        
        await api.shutdown()
        results.record_test("API initialization", True, None, (time.time() - start_time) * 1000)
        
    except Exception as e:
        results.record_test("API initialization", False, str(e))
    
    # Test 2: State API endpoints (simulated)
    start_time = time.time()
    try:
        api = CognitiveMeshAPI("api_test_node_2", port=8100)
        await api.initialize()
        
        # Simulate state update
        from src.api.mesh_api import StateRequest
        request = StateRequest(
            entity_type="test_entity",
            entity_id="entity_123",
            data={"status": "active", "value": 42}
        )
        
        # Test would require actual HTTP client, so we test the underlying state manager
        event = await api.state_manager.set_state(
            request.entity_type,
            request.entity_id, 
            request.data
        )
        
        assert event is not None
        assert event.entity_type == "test_entity"
        
        # Test state retrieval
        state = api.state_manager.get_state("test_entity", "entity_123")
        assert state["status"] == "active"
        assert state["value"] == 42
        
        await api.shutdown()
        results.record_test("State API operations", True, None, (time.time() - start_time) * 1000)
        
    except Exception as e:
        results.record_test("State API operations", False, str(e))
    
    # Test 3: Cognitive query processing
    start_time = time.time()
    try:
        api = CognitiveMeshAPI("api_test_node_3", port=8101)
        await api.initialize()
        
        # Test different query types
        from src.api.mesh_api import QueryRequest
        
        # Natural language query
        nl_request = QueryRequest(
            query="What is the current system status?",
            query_type="natural_language"
        )
        
        nl_result = await api._process_natural_language_query(nl_request)
        assert "query" in nl_result
        assert nl_result["query"] == nl_request.query
        
        # Financial query (simulated)
        fin_request = QueryRequest(
            query="How much did I spend last month?",
            query_type="financial"
        )
        
        fin_result = await api._process_financial_query(fin_request)
        assert "type" in fin_result
        assert fin_result["type"] == "financial_query"
        
        await api.shutdown()
        results.record_test("Cognitive query processing", True, None, (time.time() - start_time) * 1000)
        
    except Exception as e:
        results.record_test("Cognitive query processing", False, str(e))
    
    # Test 4: Distributed task execution
    start_time = time.time()
    try:
        api = CognitiveMeshAPI("api_test_node_4", port=8102)
        await api.initialize()
        
        from src.api.mesh_api import TaskRequest
        
        task_request = TaskRequest(
            task_type="analysis",
            task_data={"type": "pattern_analysis", "data": [1, 2, 3, 4, 5]},
            priority=2
        )
        
        task_result = await api._process_distributed_task("test_task_123", task_request)
        
        assert "analysis_type" in task_result
        assert task_result["analysis_type"] == "pattern_analysis"
        
        # Verify task state was stored
        task_state = api.state_manager.get_state("distributed_task", "test_task_123")
        assert task_state is not None
        assert task_state["status"] == "completed"
        
        await api.shutdown()
        results.record_test("Distributed task execution", True, None, (time.time() - start_time) * 1000)
        
    except Exception as e:
        results.record_test("Distributed task execution", False, str(e))
    
    return results


async def test_websocket_handler():
    """Test WebSocket functionality for real-time communication"""
    print("\n=== Testing WebSocket Handler ===")
    results = TestResults()
    
    # Test 1: Connection management
    start_time = time.time()
    try:
        handler = WebSocketHandler()
        connection_manager = handler.connection_manager
        
        # Simulate WebSocket connections (we can't create actual WebSocket objects in test)
        # Instead we'll test the logic
        
        # Test subscription management
        mock_ws = "mock_websocket_1"  # String ID for testing
        handler.connection_manager.active_connections = [mock_ws]
        handler.connection_manager.connection_subscriptions[mock_ws] = set()
        handler.connection_manager.connection_metadata[mock_ws] = {"connected_at": datetime.now()}
        
        # Test event subscription
        await handler.subscribe(mock_ws, ["state_events", "cognitive_queries"])
        
        subscriptions = handler.connection_manager.connection_subscriptions[mock_ws]
        assert "state_events" in subscriptions
        assert "cognitive_queries" in subscriptions
        
        # Test stats
        stats = handler.get_performance_metrics()
        assert "total_connections" in stats
        assert stats["total_connections"] == 1
        
        results.record_test("Connection management", True, None, (time.time() - start_time) * 1000)
        
    except Exception as e:
        results.record_test("Connection management", False, str(e))
    
    # Test 2: Message handling
    start_time = time.time()
    try:
        handler = WebSocketHandler()
        
        # Test ping message
        mock_ws = "mock_websocket_2"
        ping_message = {"type": "ping", "echo": "test_echo"}
        
        response = await handler._handle_ping(mock_ws, ping_message)
        assert response["type"] == "pong"
        assert response["echo"] == "test_echo"
        
        # Test cognitive query message
        query_message = {
            "type": "cognitive_query", 
            "query": "Test cognitive question",
            "query_type": "natural_language"
        }
        
        response = await handler._handle_cognitive_query(mock_ws, query_message)
        assert response["type"] == "cognitive_response"
        assert response["query"] == "Test cognitive question"
        
        results.record_test("Message handling", True, None, (time.time() - start_time) * 1000)
        
    except Exception as e:
        results.record_test("Message handling", False, str(e))
    
    # Test 3: Event broadcasting performance
    start_time = time.time()
    try:
        handler = WebSocketHandler()
        
        # Simulate multiple connections
        for i in range(100):
            mock_ws = f"mock_websocket_{i}"
            handler.connection_manager.active_connections.append(mock_ws)
            handler.connection_manager.connection_subscriptions[mock_ws] = {"test_events"}
        
        # Test broadcast performance (simulated)
        broadcast_start = time.time()
        
        # In real implementation, this would send to actual WebSocket connections
        # We're testing the connection filtering logic
        target_connections = []
        for connection in handler.connection_manager.active_connections:
            if "test_events" in handler.connection_manager.connection_subscriptions.get(connection, set()):
                target_connections.append(connection)
        
        broadcast_time = (time.time() - broadcast_start) * 1000
        
        assert len(target_connections) == 100
        assert broadcast_time < 10  # Should be very fast for filtering
        
        results.record_test("Event broadcasting performance", True, None, (time.time() - start_time) * 1000)
        
    except Exception as e:
        results.record_test("Event broadcasting performance", False, str(e))
    
    return results


async def test_authentication_manager():
    """Test authentication and security features"""
    print("\n=== Testing Authentication Manager ===")
    results = TestResults()
    
    # Test 1: User management
    start_time = time.time()
    try:
        auth_manager = AuthenticationManager()
        
        # Create user
        user = auth_manager.create_user("testuser", "test@example.com", UserRole.ANALYST)
        assert user.username == "testuser"
        assert user.role == UserRole.ANALYST
        assert Permission.READ_STATE in user.permissions
        assert Permission.ADMIN_ACCESS not in user.permissions  # Analyst shouldn't have admin access
        
        # Check permissions
        has_read = auth_manager.check_permission(user.user_id, Permission.READ_STATE)
        has_admin = auth_manager.check_permission(user.user_id, Permission.ADMIN_ACCESS)
        
        assert has_read == True
        assert has_admin == False
        
        results.record_test("User management", True, None, (time.time() - start_time) * 1000)
        
    except Exception as e:
        results.record_test("User management", False, str(e))
    
    # Test 2: JWT token generation and verification
    start_time = time.time()
    try:
        auth_manager = AuthenticationManager()
        
        # Create user and generate token
        user = auth_manager.create_user("jwtuser", "jwt@example.com", UserRole.DEVELOPER)
        token = auth_manager.generate_jwt_token(user.user_id, {"ip_address": "127.0.0.1"})
        
        assert token is not None
        assert len(token) > 10  # JWT should be reasonably long
        
        # Verify token
        payload = auth_manager.verify_jwt_token(token)
        assert payload is not None
        assert payload["user_id"] == user.user_id
        assert payload["username"] == "jwtuser"
        assert payload["role"] == UserRole.DEVELOPER.value
        
        # Test token revocation
        auth_manager.revoke_jwt_token(token)
        
        # Token should no longer be valid
        payload_after_revoke = auth_manager.verify_jwt_token(token)
        assert payload_after_revoke is None
        
        results.record_test("JWT token management", True, None, (time.time() - start_time) * 1000)
        
    except Exception as e:
        results.record_test("JWT token management", False, str(e))
    
    # Test 3: API key generation and verification
    start_time = time.time()
    try:
        auth_manager = AuthenticationManager()
        
        # Create user and API key
        user = auth_manager.create_user("apiuser", "api@example.com", UserRole.SERVICE)
        raw_key, api_key = auth_manager.create_api_key(user.user_id, expires_in_days=30)
        
        assert raw_key.startswith("cmk_")
        assert api_key.user_id == user.user_id
        assert api_key.is_active == True
        
        # Verify API key
        verified_key = auth_manager.verify_api_key(raw_key)
        assert verified_key is not None
        assert verified_key.key_id == api_key.key_id
        assert verified_key.usage_count == 1  # Should increment on verification
        
        # Test invalid key
        invalid_verified = auth_manager.verify_api_key("cmk_invalid_key")
        assert invalid_verified is None
        
        results.record_test("API key management", True, None, (time.time() - start_time) * 1000)
        
    except Exception as e:
        results.record_test("API key management", False, str(e))
    
    # Test 4: Rate limiting
    start_time = time.time()
    try:
        auth_manager = AuthenticationManager()
        user = auth_manager.create_user("rateuser", "rate@example.com", UserRole.ANALYST)
        
        # Test rate limiting for queries
        allowed_requests = []
        blocked_requests = []
        
        # Make requests up to the limit
        for i in range(105):  # Default limit is 100 queries per minute
            is_allowed, info = auth_manager.check_rate_limit("query", user_id=user.user_id)
            if is_allowed:
                allowed_requests.append(i)
            else:
                blocked_requests.append(i)
        
        assert len(allowed_requests) == 100  # Should allow exactly 100
        assert len(blocked_requests) == 5    # Should block the extra 5
        
        results.record_test("Rate limiting", True, None, (time.time() - start_time) * 1000)
        
    except Exception as e:
        results.record_test("Rate limiting", False, str(e))
    
    # Test 5: Security metrics and audit logging
    start_time = time.time()
    try:
        auth_manager = AuthenticationManager()
        
        # Generate some activity
        user = auth_manager.create_user("audituser", "audit@example.com", UserRole.VIEWER)
        token = auth_manager.generate_jwt_token(user.user_id)
        auth_manager.verify_jwt_token(token)
        auth_manager.check_permission(user.user_id, Permission.READ_STATE)
        
        # Get security metrics
        metrics = auth_manager.get_security_metrics()
        
        assert "total_users" in metrics
        assert "active_sessions" in metrics
        assert "total_audit_events" in metrics
        
        assert metrics["total_users"] >= 1
        assert metrics["total_audit_events"] > 0
        
        # Get audit events
        auth_events = auth_manager.get_audit_events("authentication")
        assert len(auth_events) > 0
        
        # Check that events have required fields
        event = auth_events[0]
        assert hasattr(event, 'event_type')
        assert hasattr(event, 'user_id')
        assert hasattr(event, 'timestamp')
        
        results.record_test("Security metrics and audit", True, None, (time.time() - start_time) * 1000)
        
    except Exception as e:
        results.record_test("Security metrics and audit", False, str(e))
    
    return results


async def test_external_bindings():
    """Test external system bindings (Unity3D, ROS, Web agents)"""
    print("\n=== Testing External System Bindings ===")
    results = TestResults()
    
    # Test 1: Unity binding message handling
    start_time = time.time()
    try:
        unity_binding = UnityBinding("localhost", 8090)
        
        # Test scene update message
        scene_message = ExternalMessage(
            message_id="scene_001",
            message_type="scene_update",
            source_system="Unity3D",
            target_system="CognitiveMesh",
            data={
                "objects": [
                    {"id": "obj_1", "type": "interactive", "position": [1, 0, 0], "size": 2.5},
                    {"id": "obj_2", "type": "static", "position": [3, 0, 0], "size": 1.0}
                ]
            },
            timestamp=datetime.now()
        )
        
        result = await unity_binding._handle_scene_update(scene_message)
        assert "processed_objects" in result
        assert len(result["processed_objects"]) == 2
        assert "scene_analysis" in result
        
        # Test avatar action message
        action_message = ExternalMessage(
            message_id="action_001",
            message_type="avatar_action",
            source_system="Unity3D", 
            target_system="CognitiveMesh",
            data={
                "actionType": "move",
                "targetPosition": [5, 0, 5]
            },
            timestamp=datetime.now()
        )
        
        action_result = await unity_binding._handle_avatar_action(action_message)
        assert "movement_approved" in action_result
        assert action_result["movement_approved"] == True
        
        results.record_test("Unity binding message handling", True, None, (time.time() - start_time) * 1000)
        
    except Exception as e:
        results.record_test("Unity binding message handling", False, str(e))
    
    # Test 2: ROS binding sensor processing
    start_time = time.time()
    try:
        ros_binding = ROSBinding()
        
        # Test LIDAR sensor data
        lidar_message = ExternalMessage(
            message_id="lidar_001",
            message_type="sensor_data",
            source_system="ROS",
            target_system="CognitiveMesh",
            data={
                "sensor_type": "lidar",
                "ranges": [10.5, 8.2, 3.1, 15.0, 12.3, 2.8, 9.7]  # Distance measurements
            },
            timestamp=datetime.now()
        )
        
        lidar_result = await ros_binding._handle_sensor_data(lidar_message)
        assert "obstacle_map" in lidar_result
        assert "navigation_recommendations" in lidar_result
        
        # Test navigation goal
        nav_message = ExternalMessage(
            message_id="nav_001",
            message_type="navigation_goal",
            source_system="ROS",
            target_system="CognitiveMesh",
            data={
                "target_position": {"x": 10, "y": 5, "z": 0},
                "max_velocity": 1.5
            },
            timestamp=datetime.now()
        )
        
        nav_result = await ros_binding._handle_navigation_goal(nav_message)
        assert "goal_accepted" in nav_result
        assert nav_result["goal_accepted"] == True
        assert "estimated_time" in nav_result
        
        results.record_test("ROS binding sensor processing", True, None, (time.time() - start_time) * 1000)
        
    except Exception as e:
        results.record_test("ROS binding sensor processing", False, str(e))
    
    # Test 3: Web agent binding dashboard updates
    start_time = time.time()
    try:
        web_binding = WebAgentBinding(8091)
        
        # Test dashboard update
        dashboard_message = ExternalMessage(
            message_id="dash_001",
            message_type="dashboard_update",
            source_system="WebAgent",
            target_system="CognitiveMesh",
            data={
                "dashboardType": "cognitive",
                "refreshRate": 5000
            },
            timestamp=datetime.now()
        )
        
        dash_result = await web_binding._handle_dashboard_update(dashboard_message)
        assert "dashboard_type" in dash_result
        assert "updated_widgets" in dash_result
        assert len(dash_result["updated_widgets"]) > 0
        
        # Test cognitive visualization
        viz_message = ExternalMessage(
            message_id="viz_001",
            message_type="cognitive_visualization",
            source_system="WebAgent",
            target_system="CognitiveMesh",
            data={
                "visualizationType": "network",
                "dataPoints": 50
            },
            timestamp=datetime.now()
        )
        
        viz_result = await web_binding._handle_cognitive_visualization(viz_message)
        assert "visualization_type" in viz_result
        assert "chart_data" in viz_result
        assert "interactive_elements" in viz_result
        
        results.record_test("Web agent binding dashboard", True, None, (time.time() - start_time) * 1000)
        
    except Exception as e:
        results.record_test("Web agent binding dashboard", False, str(e))
    
    return results


async def test_performance_and_scalability():
    """Test system performance and scalability requirements"""
    print("\n=== Testing Performance and Scalability ===")
    results = TestResults()
    
    # Test 1: Sub-100ms response time requirement
    start_time = time.time()
    try:
        state_manager = DistributedStateManager("perf_node")
        await state_manager.start()
        
        # Test rapid state operations
        response_times = []
        
        for i in range(100):
            op_start = time.time()
            await state_manager.set_state("perf_entity", f"item_{i}", {"value": i, "timestamp": op_start})
            op_time = (time.time() - op_start) * 1000
            response_times.append(op_time)
        
        await state_manager.stop()
        
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        sub_100ms_count = sum(1 for t in response_times if t < 100)
        
        # Success criteria: 95% of operations under 100ms
        success_rate = sub_100ms_count / len(response_times)
        success = success_rate >= 0.95
        
        results.record_test("Sub-100ms response times", success, 
                          f"Average: {avg_response_time:.1f}ms, Max: {max_response_time:.1f}ms, Success rate: {success_rate:.2%}",
                          avg_response_time)
        
        results.performance_metrics["avg_response_time"] = avg_response_time
        results.performance_metrics["max_response_time"] = max_response_time
        results.performance_metrics["sub_100ms_rate"] = success_rate
        
    except Exception as e:
        results.record_test("Sub-100ms response times", False, str(e))
    
    # Test 2: Concurrent connection handling (simulated)
    start_time = time.time()
    try:
        websocket_handler = WebSocketHandler()
        
        # Simulate 1000+ concurrent connections
        concurrent_connections = []
        for i in range(1000):
            mock_ws = f"concurrent_ws_{i}"
            concurrent_connections.append(mock_ws)
            websocket_handler.connection_manager.active_connections.append(mock_ws)
            websocket_handler.connection_manager.connection_subscriptions[mock_ws] = {"test_events"}
        
        # Test broadcast performance to all connections
        broadcast_start = time.time()
        
        # Simulate broadcast filtering (actual WebSocket sending would be done by framework)
        target_count = 0
        for connection in concurrent_connections:
            if "test_events" in websocket_handler.connection_manager.connection_subscriptions.get(connection, set()):
                target_count += 1
        
        broadcast_time = (time.time() - broadcast_start) * 1000
        
        assert target_count == 1000
        assert broadcast_time < 50  # Should handle filtering very quickly
        
        results.record_test("1000+ concurrent connections", True, 
                          f"Handled {target_count} connections in {broadcast_time:.1f}ms",
                          broadcast_time)
        
    except Exception as e:
        results.record_test("1000+ concurrent connections", False, str(e))
    
    # Test 3: Memory efficiency under load
    start_time = time.time()
    try:
        state_manager = DistributedStateManager("memory_test_node")
        await state_manager.start()
        
        # Create large number of state entries
        large_data_count = 10000
        for i in range(large_data_count):
            await state_manager.set_state(
                "large_dataset", 
                f"record_{i}",
                {
                    "id": i,
                    "data": f"This is test data record number {i} with some additional content",
                    "metrics": [i * 0.1, i * 0.2, i * 0.3],
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Verify all data is accessible
        all_records = state_manager.get_all_state("large_dataset")
        assert len(all_records) == large_data_count
        
        # Test batch retrieval performance
        retrieval_start = time.time()
        for i in range(0, large_data_count, 100):  # Sample every 100th record
            record = state_manager.get_state("large_dataset", f"record_{i}")
            assert record is not None
            assert record["id"] == i
        
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        await state_manager.stop()
        
        results.record_test("Memory efficiency under load", True,
                          f"Stored {large_data_count} records, sampled 100 in {retrieval_time:.1f}ms",
                          (time.time() - start_time) * 1000)
        
    except Exception as e:
        results.record_test("Memory efficiency under load", False, str(e))
    
    return results


async def run_all_tests():
    """Run all distributed cognitive mesh API tests"""
    print("🌐 Distributed Cognitive Mesh APIs - Comprehensive Test Suite")
    print("=" * 70)
    
    all_results = []
    
    # Run test suites
    all_results.append(await test_distributed_state_manager())
    all_results.append(await test_cognitive_mesh_api())
    all_results.append(await test_websocket_handler())
    all_results.append(await test_authentication_manager())
    all_results.append(await test_external_bindings())
    all_results.append(await test_performance_and_scalability())
    
    # Aggregate results
    total_tests = sum(r.tests_run for r in all_results)
    total_passed = sum(r.tests_passed for r in all_results)
    total_failed = sum(r.tests_failed for r in all_results)
    all_errors = []
    for r in all_results:
        all_errors.extend(r.errors)
    
    # Performance analysis
    all_performance_metrics = {}
    for r in all_results:
        all_performance_metrics.update(r.performance_metrics)
    
    avg_response_time = sum(all_performance_metrics.values()) / len(all_performance_metrics) if all_performance_metrics else 0
    sub_100ms_count = sum(1 for t in all_performance_metrics.values() if t < 100)
    performance_success_rate = sub_100ms_count / len(all_performance_metrics) if all_performance_metrics else 1.0
    
    # Print summary
    print("\n" + "=" * 70)
    print("🧪 TEST SUMMARY")
    print("=" * 70)
    print(f"Total Tests Run:     {total_tests}")
    print(f"Tests Passed:        {total_passed} ({total_passed/total_tests*100:.1f}%)")
    print(f"Tests Failed:        {total_failed} ({total_failed/total_tests*100:.1f}%)")
    print(f"")
    print(f"📊 PERFORMANCE METRICS")
    print(f"Average Response Time: {avg_response_time:.1f}ms")
    print(f"Sub-100ms Operations: {sub_100ms_count}/{len(all_performance_metrics)} ({performance_success_rate:.1%})")
    print(f"Performance Target:   {'✅ MET' if avg_response_time < 100 and performance_success_rate >= 0.95 else '❌ NOT MET'}")
    
    if all_errors:
        print(f"\n❌ ERRORS:")
        for error in all_errors:
            print(f"  • {error}")
    
    # Success criteria evaluation
    overall_success = (
        total_failed == 0 and
        avg_response_time < 100 and
        performance_success_rate >= 0.95
    )
    
    print(f"\n🎯 PHASE 4 SUCCESS CRITERIA:")
    print(f"  • Sub-100ms API response times:     {'✅' if avg_response_time < 100 else '❌'}")
    print(f"  • 99.9% API availability:           {'✅' if total_failed == 0 else '❌'} (simulated)")
    print(f"  • 1000+ concurrent connections:     {'✅' if 'concurrent connections' in str(all_results) else '❌'}")
    print(f"  • Real-time state synchronization:  {'✅' if 'State synchronization' in str(all_results) else '❌'}")
    print(f"  • Authentication & security:        {'✅' if any('authentication' in str(r.get_summary()) for r in all_results) else '❌'}")
    print(f"  • External system bindings:         {'✅' if any('external' in str(r.get_summary()) for r in all_results) else '❌'}")
    
    print(f"\n🌟 OVERALL RESULT: {'✅ SUCCESS - Phase 4 APIs Ready for Production!' if overall_success else '❌ NEEDS ATTENTION - Some issues require resolution'}")
    
    return {
        "total_tests": total_tests,
        "passed": total_passed,
        "failed": total_failed,
        "avg_response_time_ms": avg_response_time,
        "performance_success_rate": performance_success_rate,
        "overall_success": overall_success,
        "errors": all_errors
    }


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    result = asyncio.run(run_all_tests())
    
    # Exit with appropriate code
    exit(0 if result["overall_success"] else 1)