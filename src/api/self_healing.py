"""
Self-Healing Distributed System for Cognitive Mesh

Provides automatic recovery and self-healing capabilities including:
- Health monitoring and anomaly detection
- Automatic failover and recovery
- Circuit breaker pattern implementation
- Load balancing and rebalancing
- Graceful degradation
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import statistics


class HealthStatus(Enum):
    """Node health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class HealthMetrics:
    """Health metrics for a node or service"""
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    success_count: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    memory_usage_percent: float = 0.0
    cpu_usage_percent: float = 0.0
    connection_count: int = 0
    
    @property
    def average_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    @property
    def p95_response_time(self) -> float:
        if len(self.response_times) < 2:
            return self.average_response_time
        return statistics.quantiles(self.response_times, n=20)[18]  # 95th percentile
    
    @property
    def error_rate(self) -> float:
        total = self.error_count + self.success_count
        if total == 0:
            return 0.0
        return self.error_count / total


@dataclass
class HealthCheck:
    """Health check configuration"""
    check_id: str
    name: str
    check_fn: Callable
    interval_seconds: float = 30.0
    timeout_seconds: float = 10.0
    failure_threshold: int = 3
    success_threshold: int = 2
    is_critical: bool = False


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0
    half_open_max_calls: int = 3
    error_rate_threshold: float = 0.5


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance
    
    Prevents cascading failures by temporarily blocking calls to failing services.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time: Optional[datetime] = None
        self._state_changed_at: datetime = datetime.now()
        
        self.logger = logging.getLogger(__name__)
    
    def can_execute(self) -> bool:
        """Check if operation can be executed"""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self._should_attempt_reset():
                self._transition_to(CircuitState.HALF_OPEN)
                return True
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            # Allow limited calls for testing
            return self._half_open_calls < self.config.half_open_max_calls
        
        return False
    
    def record_success(self):
        """Record successful operation"""
        self._success_count += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            if self._half_open_calls >= self.config.half_open_max_calls:
                # Recovered - close circuit
                self._transition_to(CircuitState.CLOSED)
                self.logger.info(f"Circuit {self.name} recovered, transitioning to CLOSED")
        
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0
    
    def record_failure(self, error: str = None):
        """Record failed operation"""
        self._failure_count += 1
        self._last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            # Failed during recovery test - reopen circuit
            self._transition_to(CircuitState.OPEN)
            self.logger.warning(f"Circuit {self.name} failed during recovery, reopening")
        
        elif self.state == CircuitState.CLOSED:
            if self._failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)
                self.logger.warning(f"Circuit {self.name} opened due to failures: {error}")
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if not self._last_failure_time:
            return True
        
        elapsed = (datetime.now() - self._state_changed_at).total_seconds()
        return elapsed >= self.config.recovery_timeout_seconds
    
    def _transition_to(self, new_state: CircuitState):
        """Transition to new state"""
        old_state = self.state
        self.state = new_state
        self._state_changed_at = datetime.now()
        
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
        
        self.logger.info(f"Circuit {self.name}: {old_state.value} -> {new_state.value}")
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get circuit breaker state information"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time.isoformat() if self._last_failure_time else None,
            "state_changed_at": self._state_changed_at.isoformat()
        }


class HealthMonitor:
    """
    Health monitoring system for distributed nodes
    
    Features:
    - Periodic health checks
    - Anomaly detection
    - Health status aggregation
    - Alert generation
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        
        # Health checks
        self._health_checks: Dict[str, HealthCheck] = {}
        self._check_results: Dict[str, List[bool]] = {}
        
        # Node health tracking
        self._node_health: Dict[str, HealthMetrics] = {}
        self._node_status: Dict[str, HealthStatus] = {}
        
        # Monitoring
        self._monitor_task: Optional[asyncio.Task] = None
        self._alert_handlers: List[Callable] = []
        
        # Thresholds
        self.response_time_threshold_ms = 100.0
        self.error_rate_threshold = 0.1
        self.memory_threshold_percent = 90.0
        self.cpu_threshold_percent = 90.0
        
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start health monitoring"""
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info(f"Health monitor started on node {self.node_id}")
    
    async def stop(self):
        """Stop health monitoring"""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Health monitor stopped")
    
    def register_health_check(self, check: HealthCheck):
        """Register a health check"""
        self._health_checks[check.check_id] = check
        self._check_results[check.check_id] = []
        self.logger.debug(f"Registered health check: {check.name}")
    
    def register_alert_handler(self, handler: Callable):
        """Register an alert handler"""
        self._alert_handlers.append(handler)
    
    def record_node_health(self, node_id: str, metrics: Dict[str, Any]):
        """Record health metrics for a node"""
        if node_id not in self._node_health:
            self._node_health[node_id] = HealthMetrics()
        
        health = self._node_health[node_id]
        
        if "response_time_ms" in metrics:
            health.response_times.append(metrics["response_time_ms"])
        
        if "success" in metrics:
            if metrics["success"]:
                health.success_count += 1
                health.last_success_time = datetime.now()
            else:
                health.error_count += 1
                health.last_error = metrics.get("error", "Unknown error")
                health.last_error_time = datetime.now()
        
        if "memory_usage_percent" in metrics:
            health.memory_usage_percent = metrics["memory_usage_percent"]
        
        if "cpu_usage_percent" in metrics:
            health.cpu_usage_percent = metrics["cpu_usage_percent"]
        
        if "connection_count" in metrics:
            health.connection_count = metrics["connection_count"]
        
        # Update status
        self._update_node_status(node_id)
    
    def _update_node_status(self, node_id: str):
        """Update health status for a node"""
        health = self._node_health.get(node_id)
        if not health:
            self._node_status[node_id] = HealthStatus.UNKNOWN
            return
        
        # Check for unhealthy conditions
        is_unhealthy = (
            health.error_rate > self.error_rate_threshold * 2 or
            health.average_response_time > self.response_time_threshold_ms * 5 or
            health.memory_usage_percent > self.memory_threshold_percent or
            health.cpu_usage_percent > self.cpu_threshold_percent
        )
        
        if is_unhealthy:
            old_status = self._node_status.get(node_id)
            self._node_status[node_id] = HealthStatus.UNHEALTHY
            
            if old_status != HealthStatus.UNHEALTHY:
                self._trigger_alert(node_id, "unhealthy", f"Node {node_id} is unhealthy")
            return
        
        # Check for degraded conditions
        is_degraded = (
            health.error_rate > self.error_rate_threshold or
            health.average_response_time > self.response_time_threshold_ms * 2 or
            health.memory_usage_percent > self.memory_threshold_percent * 0.8 or
            health.cpu_usage_percent > self.cpu_threshold_percent * 0.8
        )
        
        if is_degraded:
            self._node_status[node_id] = HealthStatus.DEGRADED
        else:
            self._node_status[node_id] = HealthStatus.HEALTHY
    
    def _trigger_alert(self, node_id: str, alert_type: str, message: str):
        """Trigger an alert"""
        alert = {
            "node_id": node_id,
            "alert_type": alert_type,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.warning(f"Alert: {message}")
        
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                # Run all health checks
                for check_id, check in self._health_checks.items():
                    try:
                        result = await asyncio.wait_for(
                            check.check_fn(),
                            timeout=check.timeout_seconds
                        )
                        self._check_results[check_id].append(result)
                        
                        # Keep only recent results
                        if len(self._check_results[check_id]) > check.failure_threshold * 2:
                            self._check_results[check_id] = self._check_results[check_id][-check.failure_threshold * 2:]
                        
                        # Check for failures
                        recent = self._check_results[check_id][-check.failure_threshold:]
                        if len(recent) >= check.failure_threshold and all(not r for r in recent):
                            self._trigger_alert(
                                self.node_id,
                                "health_check_failed",
                                f"Health check '{check.name}' failed {check.failure_threshold} times"
                            )
                    
                    except asyncio.TimeoutError:
                        self._check_results[check_id].append(False)
                        self.logger.warning(f"Health check {check.name} timed out")
                    
                    except Exception as e:
                        self._check_results[check_id].append(False)
                        self.logger.error(f"Health check {check.name} failed: {e}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    def get_node_status(self, node_id: str) -> HealthStatus:
        """Get health status for a node"""
        return self._node_status.get(node_id, HealthStatus.UNKNOWN)
    
    def get_all_statuses(self) -> Dict[str, HealthStatus]:
        """Get all node health statuses"""
        return dict(self._node_status)
    
    def get_health_metrics(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get health metrics for a node"""
        health = self._node_health.get(node_id)
        if not health:
            return None
        
        return {
            "node_id": node_id,
            "status": self._node_status.get(node_id, HealthStatus.UNKNOWN).value,
            "average_response_time_ms": health.average_response_time,
            "p95_response_time_ms": health.p95_response_time,
            "error_rate": health.error_rate,
            "success_count": health.success_count,
            "error_count": health.error_count,
            "memory_usage_percent": health.memory_usage_percent,
            "cpu_usage_percent": health.cpu_usage_percent,
            "connection_count": health.connection_count,
            "last_error": health.last_error,
            "last_error_time": health.last_error_time.isoformat() if health.last_error_time else None
        }


class LoadBalancer:
    """
    Intelligent load balancer for distributed cognitive mesh
    
    Features:
    - Multiple balancing strategies
    - Health-aware routing
    - Adaptive weight adjustment
    - Sticky sessions support
    """
    
    def __init__(self, health_monitor: HealthMonitor):
        self.health_monitor = health_monitor
        
        # Node weights and state
        self._node_weights: Dict[str, float] = {}
        self._node_connections: Dict[str, int] = {}
        self._sticky_sessions: Dict[str, str] = {}  # session_id -> node_id
        
        # Round-robin state
        self._rr_index = 0
        self._rr_nodes: List[str] = []
        
        self.logger = logging.getLogger(__name__)
    
    def register_node(self, node_id: str, weight: float = 1.0):
        """Register a node with optional weight"""
        self._node_weights[node_id] = weight
        self._node_connections[node_id] = 0
        self._rr_nodes.append(node_id)
        self.logger.info(f"Registered node {node_id} with weight {weight}")
    
    def deregister_node(self, node_id: str):
        """Deregister a node"""
        if node_id in self._node_weights:
            del self._node_weights[node_id]
        if node_id in self._node_connections:
            del self._node_connections[node_id]
        if node_id in self._rr_nodes:
            self._rr_nodes.remove(node_id)
        
        # Remove sticky sessions pointing to this node
        sessions_to_remove = [s for s, n in self._sticky_sessions.items() if n == node_id]
        for session in sessions_to_remove:
            del self._sticky_sessions[session]
    
    def select_node_round_robin(self) -> Optional[str]:
        """Select node using round-robin strategy"""
        healthy_nodes = self._get_healthy_nodes()
        if not healthy_nodes:
            return None
        
        # Find next healthy node in round-robin
        for _ in range(len(self._rr_nodes)):
            node = self._rr_nodes[self._rr_index % len(self._rr_nodes)]
            self._rr_index += 1
            
            if node in healthy_nodes:
                return node
        
        return None
    
    def select_node_least_connections(self) -> Optional[str]:
        """Select node with least active connections"""
        healthy_nodes = self._get_healthy_nodes()
        if not healthy_nodes:
            return None
        
        # Find node with minimum connections
        min_conn = float('inf')
        selected = None
        
        for node_id in healthy_nodes:
            conn = self._node_connections.get(node_id, 0)
            if conn < min_conn:
                min_conn = conn
                selected = node_id
        
        return selected
    
    def select_node_weighted(self) -> Optional[str]:
        """Select node based on weights (higher weight = more traffic)"""
        healthy_nodes = self._get_healthy_nodes()
        if not healthy_nodes:
            return None
        
        # Calculate weighted scores
        import random
        weighted_choices = []
        for node_id in healthy_nodes:
            weight = self._node_weights.get(node_id, 1.0)
            # Adjust weight based on current load
            conn = self._node_connections.get(node_id, 0)
            adjusted_weight = weight / (1 + conn * 0.1)
            weighted_choices.append((node_id, adjusted_weight))
        
        # Select based on weights
        total_weight = sum(w for _, w in weighted_choices)
        r = random.uniform(0, total_weight)
        
        cumulative = 0
        for node_id, weight in weighted_choices:
            cumulative += weight
            if r <= cumulative:
                return node_id
        
        return weighted_choices[-1][0] if weighted_choices else None
    
    def select_node_adaptive(self) -> Optional[str]:
        """Select node using adaptive strategy based on health metrics"""
        healthy_nodes = self._get_healthy_nodes()
        if not healthy_nodes:
            return None
        
        # Score each node
        scored_nodes = []
        for node_id in healthy_nodes:
            metrics = self.health_monitor.get_health_metrics(node_id)
            
            if metrics:
                # Lower score is better
                score = (
                    metrics.get("average_response_time_ms", 100) * 0.4 +
                    metrics.get("error_rate", 0) * 1000 * 0.3 +
                    metrics.get("cpu_usage_percent", 50) * 0.2 +
                    self._node_connections.get(node_id, 0) * 10 * 0.1
                )
            else:
                score = 100  # Default score for unknown nodes
            
            scored_nodes.append((node_id, score))
        
        # Select node with best (lowest) score
        scored_nodes.sort(key=lambda x: x[1])
        return scored_nodes[0][0] if scored_nodes else None
    
    def select_node_sticky(self, session_id: str) -> Optional[str]:
        """Select node with sticky session support"""
        # Check if session already has a node
        if session_id in self._sticky_sessions:
            node_id = self._sticky_sessions[session_id]
            
            # Check if node is still healthy
            if node_id in self._get_healthy_nodes():
                return node_id
            
            # Node unhealthy - remove sticky session
            del self._sticky_sessions[session_id]
        
        # Select new node
        node_id = self.select_node_adaptive()
        if node_id:
            self._sticky_sessions[session_id] = node_id
        
        return node_id
    
    def _get_healthy_nodes(self) -> Set[str]:
        """Get set of healthy nodes"""
        healthy = set()
        for node_id in self._node_weights.keys():
            status = self.health_monitor.get_node_status(node_id)
            if status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
                healthy.add(node_id)
        return healthy
    
    def increment_connections(self, node_id: str):
        """Increment connection count for a node"""
        self._node_connections[node_id] = self._node_connections.get(node_id, 0) + 1
    
    def decrement_connections(self, node_id: str):
        """Decrement connection count for a node"""
        if node_id in self._node_connections:
            self._node_connections[node_id] = max(0, self._node_connections[node_id] - 1)
    
    def update_node_weight(self, node_id: str, weight: float):
        """Update weight for a node"""
        if node_id in self._node_weights:
            self._node_weights[node_id] = weight
    
    def get_load_info(self) -> Dict[str, Any]:
        """Get load balancer information"""
        return {
            "nodes": {
                node_id: {
                    "weight": self._node_weights.get(node_id, 0),
                    "connections": self._node_connections.get(node_id, 0),
                    "status": self.health_monitor.get_node_status(node_id).value
                }
                for node_id in self._node_weights.keys()
            },
            "sticky_sessions": len(self._sticky_sessions),
            "total_connections": sum(self._node_connections.values())
        }


class SelfHealingSystem:
    """
    Self-healing system coordinator
    
    Orchestrates health monitoring, circuit breakers, and load balancing
    for automatic recovery from failures.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        
        # Components
        self.health_monitor = HealthMonitor(node_id)
        self.load_balancer = LoadBalancer(self.health_monitor)
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Recovery state
        self._recovery_tasks: Dict[str, asyncio.Task] = {}
        self._recovery_handlers: Dict[str, Callable] = {}
        
        # Coordinator task
        self._coordinator_task: Optional[asyncio.Task] = None
        
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start self-healing system"""
        await self.health_monitor.start()
        self._coordinator_task = asyncio.create_task(self._coordination_loop())
        self.logger.info(f"Self-healing system started on node {self.node_id}")
    
    async def stop(self):
        """Stop self-healing system"""
        await self.health_monitor.stop()
        
        if self._coordinator_task:
            self._coordinator_task.cancel()
            try:
                await self._coordinator_task
            except asyncio.CancelledError:
                pass
        
        # Cancel recovery tasks
        for task in self._recovery_tasks.values():
            task.cancel()
        
        self.logger.info("Self-healing system stopped")
    
    def get_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get or create a circuit breaker"""
        if name not in self._circuit_breakers:
            self._circuit_breakers[name] = CircuitBreaker(name, config)
        return self._circuit_breakers[name]
    
    def register_recovery_handler(self, failure_type: str, handler: Callable):
        """Register a recovery handler for a failure type"""
        self._recovery_handlers[failure_type] = handler
        self.logger.debug(f"Registered recovery handler for: {failure_type}")
    
    async def trigger_recovery(self, node_id: str, failure_type: str, context: Dict[str, Any] = None):
        """Trigger recovery for a failure"""
        recovery_key = f"{node_id}:{failure_type}"
        
        # Check if recovery already in progress
        if recovery_key in self._recovery_tasks and not self._recovery_tasks[recovery_key].done():
            self.logger.debug(f"Recovery already in progress for {recovery_key}")
            return
        
        handler = self._recovery_handlers.get(failure_type)
        if not handler:
            self.logger.warning(f"No recovery handler for failure type: {failure_type}")
            return
        
        # Start recovery task
        self._recovery_tasks[recovery_key] = asyncio.create_task(
            self._execute_recovery(node_id, failure_type, handler, context or {})
        )
    
    async def _execute_recovery(self, node_id: str, failure_type: str, 
                               handler: Callable, context: Dict[str, Any]):
        """Execute recovery handler"""
        self.logger.info(f"Starting recovery for {node_id}: {failure_type}")
        
        try:
            await handler(node_id, context)
            self.logger.info(f"Recovery completed for {node_id}: {failure_type}")
            
        except Exception as e:
            self.logger.error(f"Recovery failed for {node_id}: {e}")
    
    async def _coordination_loop(self):
        """Coordination loop for self-healing actions"""
        while True:
            try:
                # Check all circuit breakers
                for name, cb in self._circuit_breakers.items():
                    if cb.state == CircuitState.OPEN:
                        # Trigger recovery if not already in progress
                        await self.trigger_recovery(
                            self.node_id,
                            f"circuit_open:{name}",
                            {"circuit_name": name}
                        )
                
                # Check node health and trigger recovery
                all_statuses = self.health_monitor.get_all_statuses()
                for node_id, status in all_statuses.items():
                    if status == HealthStatus.UNHEALTHY:
                        # Trigger node recovery
                        await self.trigger_recovery(
                            node_id,
                            "node_unhealthy",
                            {"status": status.value}
                        )
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in coordination loop: {e}")
                await asyncio.sleep(5)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        return {
            "node_id": self.node_id,
            "health_statuses": {k: v.value for k, v in self.health_monitor.get_all_statuses().items()},
            "circuit_breakers": {
                name: cb.get_state_info()
                for name, cb in self._circuit_breakers.items()
            },
            "load_balancer": self.load_balancer.get_load_info(),
            "active_recoveries": len([t for t in self._recovery_tasks.values() if not t.done()]),
            "timestamp": datetime.now().isoformat()
        }
    
    async def execute_with_circuit_breaker(self, circuit_name: str, 
                                          operation: Callable,
                                          fallback: Callable = None) -> Any:
        """Execute operation with circuit breaker protection"""
        cb = self.get_circuit_breaker(circuit_name)
        
        if not cb.can_execute():
            if fallback:
                return await fallback()
            raise Exception(f"Circuit {circuit_name} is open")
        
        try:
            result = await operation()
            cb.record_success()
            return result
            
        except Exception as e:
            cb.record_failure(str(e))
            
            if fallback:
                return await fallback()
            raise
