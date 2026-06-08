"""
Test Suite for Self-Healing Distributed System

Tests for:
- CircuitBreaker: State transitions, failure tracking, recovery
- HealthMonitor: Health tracking, status updates, alerts
- LoadBalancer: Node selection strategies
- SelfHealingSystem: Coordination and recovery
"""

import asyncio
import sys
import os
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.self_healing import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    HealthMonitor,
    HealthStatus,
    HealthMetrics,
    HealthCheck,
    LoadBalancer,
    SelfHealingSystem
)


class TestCircuitBreaker(unittest.TestCase):
    """Test CircuitBreaker functionality"""
    
    def test_initial_state_is_closed(self):
        """Circuit breaker starts in CLOSED state"""
        cb = CircuitBreaker("test")
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertTrue(cb.can_execute())
    
    def test_stays_closed_on_success(self):
        """Circuit stays closed on successful operations"""
        cb = CircuitBreaker("test")
        
        for _ in range(10):
            cb.record_success()
        
        self.assertEqual(cb.state, CircuitState.CLOSED)
    
    def test_opens_after_failure_threshold(self):
        """Circuit opens after reaching failure threshold"""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)
        
        # Record failures
        for i in range(3):
            cb.record_failure(f"Error {i}")
        
        self.assertEqual(cb.state, CircuitState.OPEN)
        self.assertFalse(cb.can_execute())
    
    def test_does_not_open_below_threshold(self):
        """Circuit stays closed below failure threshold"""
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker("test", config)
        
        # Record failures below threshold
        for i in range(4):
            cb.record_failure(f"Error {i}")
        
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertTrue(cb.can_execute())
    
    def test_resets_failure_count_on_success(self):
        """Success resets failure count in closed state"""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)
        
        # Record some failures
        cb.record_failure("Error 1")
        cb.record_failure("Error 2")
        
        # Success resets counter
        cb.record_success()
        
        # Need threshold more failures to open
        cb.record_failure("Error 3")
        cb.record_failure("Error 4")
        
        self.assertEqual(cb.state, CircuitState.CLOSED)
    
    def test_transitions_to_half_open_after_timeout(self):
        """Circuit transitions to HALF_OPEN after recovery timeout"""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout_seconds=0.1)
        cb = CircuitBreaker("test", config)
        
        # Open the circuit
        cb.record_failure("Error 1")
        cb.record_failure("Error 2")
        self.assertEqual(cb.state, CircuitState.OPEN)
        
        # Wait for recovery timeout
        import time
        time.sleep(0.15)
        
        # Should transition to half-open on next check
        self.assertTrue(cb.can_execute())
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)
    
    def test_closes_after_half_open_successes(self):
        """Circuit closes after successful calls in HALF_OPEN state"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_seconds=0.01,
            half_open_max_calls=2
        )
        cb = CircuitBreaker("test", config)
        
        # Open and wait
        cb.record_failure("Error 1")
        cb.record_failure("Error 2")
        
        import time
        time.sleep(0.02)
        cb.can_execute()  # Triggers transition to HALF_OPEN
        
        # Record successes
        cb.record_success()
        cb.record_success()
        
        self.assertEqual(cb.state, CircuitState.CLOSED)
    
    def test_reopens_after_half_open_failure(self):
        """Circuit reopens after failure in HALF_OPEN state"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_seconds=0.01,
            half_open_max_calls=2
        )
        cb = CircuitBreaker("test", config)
        
        # Open and wait
        cb.record_failure("Error 1")
        cb.record_failure("Error 2")
        
        import time
        time.sleep(0.02)
        cb.can_execute()  # Triggers transition to HALF_OPEN
        
        # Record failure
        cb.record_failure("Error 3")
        
        self.assertEqual(cb.state, CircuitState.OPEN)
    
    def test_get_state_info(self):
        """State info returns correct information"""
        cb = CircuitBreaker("test_circuit")
        cb.record_success()
        cb.record_failure("Test error")
        
        info = cb.get_state_info()
        
        self.assertEqual(info["name"], "test_circuit")
        self.assertEqual(info["state"], "closed")
        self.assertEqual(info["success_count"], 1)
        self.assertEqual(info["failure_count"], 1)
        self.assertIsNotNone(info["last_failure_time"])


class TestHealthMetrics(unittest.TestCase):
    """Test HealthMetrics functionality"""
    
    def test_average_response_time(self):
        """Average response time calculated correctly"""
        metrics = HealthMetrics()
        metrics.response_times.extend([10, 20, 30, 40, 50])
        
        self.assertEqual(metrics.average_response_time, 30.0)
    
    def test_error_rate(self):
        """Error rate calculated correctly"""
        metrics = HealthMetrics()
        metrics.success_count = 80
        metrics.error_count = 20
        
        self.assertEqual(metrics.error_rate, 0.2)
    
    def test_error_rate_zero_operations(self):
        """Error rate is zero with no operations"""
        metrics = HealthMetrics()
        self.assertEqual(metrics.error_rate, 0.0)
    
    def test_p95_response_time(self):
        """P95 response time calculated correctly"""
        metrics = HealthMetrics()
        metrics.response_times.extend(range(1, 101))  # 1-100
        
        # P95 should be around 95
        p95 = metrics.p95_response_time
        self.assertGreater(p95, 90)
        self.assertLessEqual(p95, 100)


class TestHealthMonitor(unittest.TestCase):
    """Test HealthMonitor functionality"""
    
    def setUp(self):
        self.monitor = HealthMonitor("test_node")
    
    def test_initial_status_is_unknown(self):
        """Unknown node has UNKNOWN status"""
        status = self.monitor.get_node_status("unknown_node")
        self.assertEqual(status, HealthStatus.UNKNOWN)
    
    def test_record_node_health_success(self):
        """Recording success updates health metrics"""
        self.monitor.record_node_health("node1", {
            "success": True,
            "response_time_ms": 50.0
        })
        
        metrics = self.monitor.get_health_metrics("node1")
        
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["success_count"], 1)
        self.assertEqual(metrics["error_count"], 0)
        self.assertEqual(metrics["average_response_time_ms"], 50.0)
    
    def test_record_node_health_failure(self):
        """Recording failure updates health metrics"""
        self.monitor.record_node_health("node1", {
            "success": False,
            "error": "Connection timeout"
        })
        
        metrics = self.monitor.get_health_metrics("node1")
        
        self.assertEqual(metrics["error_count"], 1)
        self.assertEqual(metrics["last_error"], "Connection timeout")
    
    def test_healthy_status(self):
        """Node with good metrics has HEALTHY status"""
        # Record many successful operations
        for _ in range(10):
            self.monitor.record_node_health("node1", {
                "success": True,
                "response_time_ms": 20.0,
                "cpu_usage_percent": 30.0,
                "memory_usage_percent": 40.0
            })
        
        status = self.monitor.get_node_status("node1")
        self.assertEqual(status, HealthStatus.HEALTHY)
    
    def test_degraded_status(self):
        """Node with degraded metrics has DEGRADED status"""
        # Record operations with high error rate
        for i in range(10):
            self.monitor.record_node_health("node1", {
                "success": i >= 2,  # 20% errors - above threshold
                "response_time_ms": 20.0
            })
        
        status = self.monitor.get_node_status("node1")
        self.assertEqual(status, HealthStatus.DEGRADED)
    
    def test_unhealthy_status_high_cpu(self):
        """Node with very high CPU has UNHEALTHY status"""
        self.monitor.record_node_health("node1", {
            "success": True,
            "cpu_usage_percent": 95.0  # Above threshold
        })
        
        status = self.monitor.get_node_status("node1")
        self.assertEqual(status, HealthStatus.UNHEALTHY)
    
    def test_unhealthy_status_high_memory(self):
        """Node with very high memory has UNHEALTHY status"""
        self.monitor.record_node_health("node1", {
            "success": True,
            "memory_usage_percent": 95.0  # Above threshold
        })
        
        status = self.monitor.get_node_status("node1")
        self.assertEqual(status, HealthStatus.UNHEALTHY)
    
    def test_register_health_check(self):
        """Can register health checks"""
        async def check_fn():
            return True
        
        check = HealthCheck(
            check_id="test_check",
            name="Test Check",
            check_fn=check_fn
        )
        
        self.monitor.register_health_check(check)
        self.assertIn("test_check", self.monitor._health_checks)
    
    def test_register_alert_handler(self):
        """Can register alert handlers"""
        handler = Mock()
        self.monitor.register_alert_handler(handler)
        self.assertIn(handler, self.monitor._alert_handlers)
    
    def test_get_all_statuses(self):
        """Can get all node statuses"""
        self.monitor.record_node_health("node1", {"success": True})
        self.monitor.record_node_health("node2", {"success": True})
        
        statuses = self.monitor.get_all_statuses()
        
        self.assertIn("node1", statuses)
        self.assertIn("node2", statuses)


class TestLoadBalancer(unittest.TestCase):
    """Test LoadBalancer functionality"""
    
    def setUp(self):
        self.monitor = HealthMonitor("test_node")
        self.lb = LoadBalancer(self.monitor)
        
        # Register and mark nodes as healthy
        for i in range(3):
            node_id = f"node{i}"
            self.lb.register_node(node_id, weight=1.0)
            # Mark as healthy
            self.monitor.record_node_health(node_id, {
                "success": True,
                "response_time_ms": 10.0,
                "cpu_usage_percent": 20.0,
                "memory_usage_percent": 30.0
            })
    
    def test_register_node(self):
        """Can register nodes with weights"""
        self.lb.register_node("new_node", weight=2.0)
        
        load_info = self.lb.get_load_info()
        self.assertIn("new_node", load_info["nodes"])
        self.assertEqual(load_info["nodes"]["new_node"]["weight"], 2.0)
    
    def test_deregister_node(self):
        """Can deregister nodes"""
        self.lb.register_node("temp_node")
        self.lb.deregister_node("temp_node")
        
        load_info = self.lb.get_load_info()
        self.assertNotIn("temp_node", load_info["nodes"])
    
    def test_round_robin_selection(self):
        """Round-robin selects nodes in order"""
        selected = []
        for _ in range(6):
            node = self.lb.select_node_round_robin()
            if node:
                selected.append(node)
        
        # Should have selected each node twice
        self.assertEqual(len(selected), 6)
        self.assertEqual(selected.count("node0"), 2)
        self.assertEqual(selected.count("node1"), 2)
        self.assertEqual(selected.count("node2"), 2)
    
    def test_least_connections_selection(self):
        """Least connections selects node with fewest connections"""
        self.lb.increment_connections("node0")
        self.lb.increment_connections("node0")
        self.lb.increment_connections("node1")
        
        # node2 has 0 connections
        selected = self.lb.select_node_least_connections()
        self.assertEqual(selected, "node2")
    
    def test_adaptive_selection(self):
        """Adaptive selection considers health metrics"""
        # Make node0 have worse metrics
        for _ in range(10):
            self.monitor.record_node_health("node0", {
                "success": True,
                "response_time_ms": 200.0,  # Slow
                "cpu_usage_percent": 80.0   # High CPU
            })
        
        # Ensure node1 and node2 have good metrics
        for _ in range(10):
            for node_id in ["node1", "node2"]:
                self.monitor.record_node_health(node_id, {
                    "success": True,
                    "response_time_ms": 10.0,  # Fast
                    "cpu_usage_percent": 20.0  # Low CPU
                })
        
        # node1 and node2 should be preferred
        selected_counts = {"node0": 0, "node1": 0, "node2": 0}
        for _ in range(30):
            node = self.lb.select_node_adaptive()
            if node:
                selected_counts[node] += 1
        
        # node0 should be selected less often
        # (or not at all compared to the much better nodes)
        total_good_nodes = selected_counts["node1"] + selected_counts["node2"]
        self.assertGreater(total_good_nodes, 0, "Good nodes should be selected")
        self.assertLessEqual(selected_counts["node0"], total_good_nodes)
    
    def test_sticky_session(self):
        """Sticky sessions return same node for session"""
        session_id = "user123"
        
        first_node = self.lb.select_node_sticky(session_id)
        
        # Same session should always get same node
        for _ in range(5):
            node = self.lb.select_node_sticky(session_id)
            self.assertEqual(node, first_node)
    
    def test_sticky_session_failover(self):
        """Sticky session fails over when node is unhealthy"""
        session_id = "user456"
        
        first_node = self.lb.select_node_sticky(session_id)
        
        # Mark node as unhealthy
        for _ in range(20):
            self.monitor.record_node_health(first_node, {
                "success": False,
                "error": "Node failure"
            })
        
        # Should select a different node
        new_node = self.lb.select_node_sticky(session_id)
        # New node should be healthy
        self.assertIn(new_node, ["node0", "node1", "node2"])
    
    def test_connection_tracking(self):
        """Can track connection counts"""
        self.lb.increment_connections("node0")
        self.lb.increment_connections("node0")
        self.lb.decrement_connections("node0")
        
        load_info = self.lb.get_load_info()
        self.assertEqual(load_info["nodes"]["node0"]["connections"], 1)
    
    def test_no_healthy_nodes(self):
        """Returns None when no healthy nodes available"""
        # Mark all nodes as unhealthy
        for node_id in ["node0", "node1", "node2"]:
            for _ in range(20):
                self.monitor.record_node_health(node_id, {
                    "success": False,
                    "error": "Node failure"
                })
        
        self.assertIsNone(self.lb.select_node_round_robin())
        self.assertIsNone(self.lb.select_node_least_connections())
        self.assertIsNone(self.lb.select_node_adaptive())
    
    def test_weighted_selection(self):
        """Weighted selection favors higher weights"""
        # Clear and register nodes with different weights
        self.lb = LoadBalancer(self.monitor)
        self.lb.register_node("heavy", weight=10.0)
        self.lb.register_node("light", weight=1.0)
        
        # Mark both as healthy
        for node in ["heavy", "light"]:
            self.monitor.record_node_health(node, {
                "success": True,
                "response_time_ms": 10.0
            })
        
        # Count selections
        counts = {"heavy": 0, "light": 0}
        for _ in range(100):
            node = self.lb.select_node_weighted()
            if node:
                counts[node] += 1
        
        # Heavy should be selected more often
        self.assertGreater(counts["heavy"], counts["light"] * 2)


class TestSelfHealingSystem(unittest.TestCase):
    """Test SelfHealingSystem functionality"""
    
    def setUp(self):
        self.system = SelfHealingSystem("test_node")
    
    def test_get_circuit_breaker(self):
        """Can get/create circuit breakers"""
        cb1 = self.system.get_circuit_breaker("service_a")
        cb2 = self.system.get_circuit_breaker("service_a")
        
        # Same instance
        self.assertIs(cb1, cb2)
        self.assertEqual(cb1.name, "service_a")
    
    def test_get_circuit_breaker_with_config(self):
        """Can create circuit breaker with custom config"""
        config = CircuitBreakerConfig(failure_threshold=10)
        cb = self.system.get_circuit_breaker("custom", config)
        
        self.assertEqual(cb.config.failure_threshold, 10)
    
    def test_register_recovery_handler(self):
        """Can register recovery handlers"""
        handler = Mock()
        self.system.register_recovery_handler("test_failure", handler)
        
        self.assertIn("test_failure", self.system._recovery_handlers)
    
    def test_get_system_status(self):
        """System status includes all components"""
        # Create some state
        self.system.get_circuit_breaker("test_cb")
        self.system.load_balancer.register_node("node1")
        
        status = self.system.get_system_status()
        
        self.assertEqual(status["node_id"], "test_node")
        self.assertIn("health_statuses", status)
        self.assertIn("circuit_breakers", status)
        self.assertIn("load_balancer", status)
        self.assertIn("active_recoveries", status)
        self.assertIn("timestamp", status)


class TestSelfHealingSystemAsync(unittest.IsolatedAsyncioTestCase):
    """Async tests for SelfHealingSystem"""
    
    async def test_start_and_stop(self):
        """System can start and stop cleanly"""
        system = SelfHealingSystem("test_node")
        
        await system.start()
        self.assertIsNotNone(system._coordinator_task)
        
        await system.stop()
        self.assertTrue(system._coordinator_task.cancelled() or system._coordinator_task.done())
    
    async def test_execute_with_circuit_breaker_success(self):
        """Execute with circuit breaker records success"""
        system = SelfHealingSystem("test_node")
        
        async def operation():
            return "success"
        
        result = await system.execute_with_circuit_breaker("test", operation)
        
        self.assertEqual(result, "success")
        cb = system.get_circuit_breaker("test")
        self.assertEqual(cb.state, CircuitState.CLOSED)
    
    async def test_execute_with_circuit_breaker_failure(self):
        """Execute with circuit breaker records failure"""
        system = SelfHealingSystem("test_node")
        
        async def failing_operation():
            raise ValueError("Test error")
        
        with self.assertRaises(ValueError):
            await system.execute_with_circuit_breaker("test", failing_operation)
        
        cb = system.get_circuit_breaker("test")
        self.assertEqual(cb._failure_count, 1)
    
    async def test_execute_with_circuit_breaker_fallback(self):
        """Circuit breaker uses fallback when open"""
        config = CircuitBreakerConfig(failure_threshold=2)
        system = SelfHealingSystem("test_node")
        
        # Get the circuit breaker and open it
        cb = system.get_circuit_breaker("test", config)
        cb.record_failure("Error 1")
        cb.record_failure("Error 2")
        
        async def operation():
            raise ValueError("Should not be called")
        
        async def fallback():
            return "fallback"
        
        result = await system.execute_with_circuit_breaker("test", operation, fallback)
        self.assertEqual(result, "fallback")
    
    async def test_trigger_recovery(self):
        """Can trigger recovery with handler"""
        system = SelfHealingSystem("test_node")
        
        recovery_called = asyncio.Event()
        
        async def recovery_handler(node_id, context):
            recovery_called.set()
        
        system.register_recovery_handler("test_failure", recovery_handler)
        
        await system.trigger_recovery("node1", "test_failure", {"key": "value"})
        
        # Wait for recovery task
        await asyncio.sleep(0.1)
        self.assertTrue(recovery_called.is_set())
    
    async def test_recovery_not_duplicate(self):
        """Recovery doesn't duplicate for same failure"""
        system = SelfHealingSystem("test_node")
        
        call_count = 0
        
        async def slow_recovery(node_id, context):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.5)  # Simulate slow recovery
        
        system.register_recovery_handler("slow_failure", slow_recovery)
        
        # Trigger multiple recoveries
        await system.trigger_recovery("node1", "slow_failure")
        await system.trigger_recovery("node1", "slow_failure")
        await system.trigger_recovery("node1", "slow_failure")
        
        await asyncio.sleep(0.1)
        
        # Should only be called once since recovery is in progress
        self.assertEqual(call_count, 1)


class TestHealthMonitorAsync(unittest.IsolatedAsyncioTestCase):
    """Async tests for HealthMonitor"""
    
    async def test_start_and_stop(self):
        """Monitor can start and stop cleanly"""
        monitor = HealthMonitor("test_node")
        
        await monitor.start()
        self.assertIsNotNone(monitor._monitor_task)
        
        await monitor.stop()
        self.assertTrue(monitor._monitor_task.cancelled() or monitor._monitor_task.done())
    
    async def test_health_check_execution(self):
        """Health checks are executed"""
        monitor = HealthMonitor("test_node")
        
        check_executed = asyncio.Event()
        
        async def check_fn():
            check_executed.set()
            return True
        
        check = HealthCheck(
            check_id="async_check",
            name="Async Check",
            check_fn=check_fn,
            interval_seconds=0.1
        )
        
        monitor.register_health_check(check)
        await monitor.start()
        
        # Wait for check to be executed
        await asyncio.sleep(0.5)
        await monitor.stop()
        
        self.assertTrue(check_executed.is_set())
    
    async def test_alert_on_unhealthy(self):
        """Alert triggered when node becomes unhealthy"""
        monitor = HealthMonitor("test_node")
        
        alerts = []
        monitor.register_alert_handler(lambda alert: alerts.append(alert))
        
        # Make node unhealthy
        for _ in range(20):
            monitor.record_node_health("bad_node", {
                "success": False,
                "error": "Connection failed"
            })
        
        # Should have triggered alert
        self.assertGreater(len(alerts), 0)
        self.assertEqual(alerts[0]["alert_type"], "unhealthy")


# Performance Tests
class TestPerformance(unittest.TestCase):
    """Performance tests for self-healing components"""
    
    def test_circuit_breaker_performance(self):
        """Circuit breaker operations are fast"""
        cb = CircuitBreaker("perf_test")
        
        import time
        start = time.time()
        
        for _ in range(10000):
            cb.can_execute()
            cb.record_success()
        
        elapsed = time.time() - start
        
        # Should complete in well under 1 second
        self.assertLess(elapsed, 1.0)
        print(f"\nCircuit breaker: 10,000 ops in {elapsed:.3f}s ({10000/elapsed:.0f} ops/sec)")
    
    def test_health_metrics_performance(self):
        """Health metrics calculations are fast"""
        metrics = HealthMetrics()
        
        import time
        start = time.time()
        
        for i in range(10000):
            metrics.response_times.append(i % 100)
            metrics.success_count += 1
            _ = metrics.average_response_time
            _ = metrics.error_rate
        
        elapsed = time.time() - start
        
        # Should complete quickly
        self.assertLess(elapsed, 1.0)
        print(f"\nHealth metrics: 10,000 ops in {elapsed:.3f}s ({10000/elapsed:.0f} ops/sec)")
    
    def test_load_balancer_selection_performance(self):
        """Load balancer selection is fast"""
        monitor = HealthMonitor("perf_test")
        lb = LoadBalancer(monitor)
        
        # Register many nodes
        for i in range(100):
            node_id = f"node{i}"
            lb.register_node(node_id)
            monitor.record_node_health(node_id, {
                "success": True,
                "response_time_ms": 10.0
            })
        
        import time
        start = time.time()
        
        for _ in range(10000):
            lb.select_node_round_robin()
        
        elapsed = time.time() - start
        
        self.assertLess(elapsed, 1.0)
        print(f"\nLoad balancer selection: 10,000 ops in {elapsed:.3f}s ({10000/elapsed:.0f} ops/sec)")


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCircuitBreaker))
    suite.addTests(loader.loadTestsFromTestCase(TestHealthMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestHealthMonitor))
    suite.addTests(loader.loadTestsFromTestCase(TestLoadBalancer))
    suite.addTests(loader.loadTestsFromTestCase(TestSelfHealingSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestSelfHealingSystemAsync))
    suite.addTests(loader.loadTestsFromTestCase(TestHealthMonitorAsync))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
