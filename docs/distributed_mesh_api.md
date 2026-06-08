# Distributed Cognitive Mesh APIs - Documentation

## Overview

The Distributed Cognitive Mesh API system provides a comprehensive framework for building distributed cognitive applications. This documentation covers the complete Phase 4 implementation including REST/WebSocket APIs, state synchronization, task orchestration, and self-healing capabilities.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Core Components](#core-components)
3. [REST API Reference](#rest-api-reference)
4. [WebSocket API Reference](#websocket-api-reference)
5. [Task Orchestration](#task-orchestration)
6. [Self-Healing System](#self-healing-system)
7. [External System Bindings](#external-system-bindings)
8. [Security & Authentication](#security--authentication)
9. [Performance Guidelines](#performance-guidelines)
10. [Examples](#examples)

---

## Getting Started

### Installation

```bash
pip install fastapi uvicorn aiohttp websockets pydantic pyjwt
```

### Quick Start

```python
import asyncio
from src.api import CognitiveMeshAPI

async def main():
    # Initialize the API
    api = CognitiveMeshAPI(node_id="primary_node", port=8000)
    await api.initialize()
    
    # Start the server
    await api.start_server()

asyncio.run(main())
```

### Configuration

```python
# Environment variables
COGNITIVE_MESH_NODE_ID=primary_node
COGNITIVE_MESH_PORT=8000
COGNITIVE_MESH_HOST=0.0.0.0
JWT_SECRET=your-secret-key
JWT_EXPIRE_HOURS=24
```

---

## Core Components

### CognitiveMeshAPI

The main API gateway that provides unified REST and WebSocket endpoints.

```python
from src.api import CognitiveMeshAPI

api = CognitiveMeshAPI(
    node_id="node_001",
    port=8000,
    host="0.0.0.0"
)
```

**Key Features:**
- Sub-100ms response times for simple operations
- Real-time WebSocket communication
- Distributed state synchronization
- Authentication and rate limiting
- API versioning and backward compatibility

### DistributedStateManager

Manages distributed cognitive state across mesh nodes.

```python
from src.api import DistributedStateManager, StateNode

# Initialize
state_manager = DistributedStateManager("node_001")
await state_manager.start()

# Set state
event = await state_manager.set_state(
    entity_type="cognitive_state",
    entity_id="thought_001",
    data={"content": "Analyzing patterns", "confidence": 0.85}
)

# Get state
state = state_manager.get_state("cognitive_state", "thought_001")

# Register nodes
node = StateNode(
    node_id="node_002",
    host="localhost",
    port=8001,
    capabilities=["reasoning", "nlp"],
    last_seen=datetime.now()
)
state_manager.register_node(node)
```

### WebSocketHandler

Handles real-time bidirectional communication.

```python
from src.api import WebSocketHandler

handler = WebSocketHandler()

# Subscribe to events
await handler.subscribe(websocket, ["state_events", "cognitive_queries"])

# Broadcast events
await handler.broadcast_event("state_update", {"entity": "thought_001"})
```

---

## REST API Reference

### Health & Status

#### GET /health

Health check endpoint.

**Response:**
```json
{
    "status": "healthy",
    "node_id": "node_001",
    "timestamp": "2025-01-15T10:30:00Z",
    "uptime": "running"
}
```

#### GET /status

Get system status and statistics.

**Response:**
```json
{
    "success": true,
    "data": {
        "node_id": "node_001",
        "total_entities": 150,
        "entity_counts": {"cognitive_state": 100, "task": 50},
        "active_nodes": 3,
        "pending_events": 5
    }
}
```

### API Version

#### GET /api/version

Get API version information.

**Response:**
```json
{
    "current_version": "1.1.0",
    "supported_versions": ["1.0", "1.0.0", "1.1.0"],
    "min_supported_version": "1.0",
    "node_id": "node_001"
}
```

### State Management

#### PUT /api/v1/state

Set or update entity state.

**Request:**
```json
{
    "entity_type": "cognitive_state",
    "entity_id": "thought_001",
    "data": {
        "content": "Deep reasoning about patterns",
        "confidence": 0.92,
        "reasoning_type": "inductive"
    },
    "priority": 2
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "event_id": "node_001_cognitive_state_thought_001_1_1705312200",
        "version": 1,
        "entity_type": "cognitive_state",
        "entity_id": "thought_001"
    },
    "execution_time_ms": 15.5,
    "node_id": "node_001"
}
```

#### GET /api/v1/state/{entity_type}/{entity_id}

Get entity state.

**Response:**
```json
{
    "success": true,
    "data": {
        "entity_type": "cognitive_state",
        "entity_id": "thought_001",
        "state": {
            "content": "Deep reasoning about patterns",
            "confidence": 0.92
        }
    }
}
```

#### DELETE /api/v1/state/{entity_type}/{entity_id}

Delete entity state.

### Cognitive Operations

#### POST /api/v1/query

Execute cognitive query.

**Request:**
```json
{
    "query": "What patterns do you see in the spending data?",
    "query_type": "financial",
    "parameters": {
        "time_range": "last_month",
        "category": "expenses"
    }
}
```

**Query Types:**
- `natural_language` - General NLP processing
- `financial` - Financial analysis and reasoning
- `reasoning` - PLN-based logical reasoning

**Response:**
```json
{
    "success": true,
    "data": {
        "type": "financial_query",
        "query": "What patterns do you see?",
        "response": "Identified 3 spending patterns...",
        "confidence": 0.85,
        "agent": "financial_chat_agent"
    },
    "execution_time_ms": 45.2
}
```

#### POST /api/v1/task

Execute distributed cognitive task.

**Request:**
```json
{
    "task_type": "analysis",
    "task_data": {
        "type": "pattern_analysis",
        "data": [1, 2, 3, 4, 5],
        "options": {"depth": "detailed"}
    },
    "target_nodes": ["node_001", "node_002"],
    "priority": 1
}
```

**Task Types:**
- `analysis` - Data analysis tasks
- `reasoning` - Logical reasoning tasks
- `synchronization` - State sync tasks

### Node Management

#### POST /api/v1/nodes/register

Register a new node in the mesh.

**Request:**
```json
{
    "node_id": "node_003",
    "host": "192.168.1.100",
    "port": 8002,
    "capabilities": ["reasoning", "financial", "nlp"]
}
```

---

## WebSocket API Reference

### Connection

Connect to the WebSocket endpoint:
```
ws://localhost:8000/ws
```

### Message Types

#### Ping/Pong

```json
// Request
{"type": "ping", "echo": "test"}

// Response
{"type": "pong", "timestamp": "2025-01-15T10:30:00Z", "echo": "test"}
```

#### Subscribe to Events

```json
// Request
{"type": "subscribe", "event_types": ["state_events", "task_updates"]}

// Response
{"type": "subscribed", "event_types": ["state_events", "task_updates"]}
```

#### Cognitive Query

```json
// Request
{
    "type": "query",
    "query": "What is the system status?",
    "query_type": "natural_language"
}

// Response
{
    "type": "query_response",
    "result": {...}
}
```

#### Real-time State Update

```json
// Request
{
    "type": "state_update",
    "entity_type": "cognitive_state",
    "entity_id": "thought_001",
    "data": {"content": "Updated thought"}
}

// Response
{"type": "state_updated", "event_id": "..."}
```

### Event Notifications

When subscribed, clients receive real-time events:

```json
{
    "type": "state_event",
    "event": {
        "event_id": "...",
        "event_type": "update",
        "entity_type": "cognitive_state",
        "entity_id": "thought_001",
        "data": {...}
    },
    "timestamp": "2025-01-15T10:30:00Z"
}
```

---

## Task Orchestration

### TaskOrchestrator

Advanced task scheduling and coordination.

```python
from src.api import TaskOrchestrator, TaskDefinition, TaskPriority

# Initialize
orchestrator = TaskOrchestrator(node_id="node_001")
await orchestrator.start()

# Register task handler
async def analysis_handler(payload):
    return {"result": "Analysis complete", "data": payload}

orchestrator.register_handler("analysis", analysis_handler)

# Submit task
task = TaskDefinition(
    task_id="task_001",
    task_type="analysis",
    payload={"data": [1, 2, 3]},
    priority=TaskPriority.HIGH,
    timeout_seconds=30.0,
    max_retries=3
)
task_id = await orchestrator.submit_task(task)

# Wait for result
result = await orchestrator.wait_for_task(task_id, timeout=60.0)
```

### Task Dependencies

```python
# Task with dependencies
dependent_task = TaskDefinition(
    task_id="task_003",
    task_type="aggregation",
    payload={"aggregate": True},
    dependencies=["task_001", "task_002"]  # Will wait for these
)
```

### DistributedTaskCoordinator

Higher-level workflow coordination.

```python
from src.api import DistributedTaskCoordinator

coordinator = DistributedTaskCoordinator(orchestrator)

# Execute workflow
results = await coordinator.execute_workflow(
    workflow_id="wf_001",
    tasks=[task1, task2, task3]
)

# Map-reduce pattern
results = await coordinator.map_reduce_task(
    task_type="analysis",
    data_chunks=[chunk1, chunk2, chunk3],
    reduce_fn=lambda r: sum(r)
)

# Scatter-gather pattern
results = await coordinator.scatter_gather(
    task_type="health_check",
    payload={"check": "status"},
    target_nodes=["node_001", "node_002", "node_003"]
)
```

---

## Self-Healing System

### SelfHealingSystem

Comprehensive self-healing capabilities.

```python
from src.api import SelfHealingSystem

# Initialize
healing = SelfHealingSystem(node_id="node_001")
await healing.start()

# Register node
healing.load_balancer.register_node("node_002", weight=1.0)

# Get circuit breaker
cb = healing.get_circuit_breaker("external_service")

# Execute with circuit breaker protection
result = await healing.execute_with_circuit_breaker(
    circuit_name="api_call",
    operation=make_api_call,
    fallback=get_cached_response
)

# Register recovery handler
async def node_recovery(node_id, context):
    # Perform recovery actions
    pass

healing.register_recovery_handler("node_unhealthy", node_recovery)
```

### HealthMonitor

Monitor health of distributed nodes.

```python
from src.api import HealthMonitor, HealthCheck

monitor = HealthMonitor(node_id="node_001")
await monitor.start()

# Register health check
async def check_database():
    # Return True if healthy
    return await db.ping()

check = HealthCheck(
    check_id="db_check",
    name="Database Health",
    check_fn=check_database,
    interval_seconds=30.0,
    failure_threshold=3,
    is_critical=True
)
monitor.register_health_check(check)

# Record metrics
monitor.record_node_health("node_002", {
    "response_time_ms": 45,
    "success": True,
    "memory_usage_percent": 65,
    "cpu_usage_percent": 30
})

# Get status
status = monitor.get_node_status("node_002")
metrics = monitor.get_health_metrics("node_002")
```

### LoadBalancer

Intelligent load balancing.

```python
from src.api import LoadBalancer

# Selection strategies
node = load_balancer.select_node_round_robin()
node = load_balancer.select_node_least_connections()
node = load_balancer.select_node_weighted()
node = load_balancer.select_node_adaptive()  # Health-aware
node = load_balancer.select_node_sticky(session_id)
```

### CircuitBreaker

Circuit breaker pattern for fault tolerance.

```python
from src.api import CircuitBreaker, CircuitBreakerConfig

config = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout_seconds=30.0,
    half_open_max_calls=3,
    error_rate_threshold=0.5
)

cb = CircuitBreaker("api_service", config)

if cb.can_execute():
    try:
        result = await make_call()
        cb.record_success()
    except Exception as e:
        cb.record_failure(str(e))
```

---

## External System Bindings

### Unity3D Binding

```python
from src.api import UnityBinding, ExternalMessage

unity = UnityBinding(host="localhost", port=8080)
await unity.connect()

# Send to Unity
message = ExternalMessage(
    message_id="msg_001",
    message_type="avatar_action",
    source_system="CognitiveMesh",
    target_system="Unity3D",
    data={"action": "wave", "speed": 1.0},
    timestamp=datetime.now()
)
await unity.send_message(message)

# Register handler for Unity events
def handle_physics(msg):
    print(f"Physics event: {msg.data}")

unity.register_handler("physics_event", handle_physics)
```

### ROS Binding

```python
from src.api import ROSBinding

ros = ROSBinding(master_uri="http://localhost:11311")
await ros.connect()

# Subscribe to sensor data
ros.register_handler("sensor_data", process_sensor)
ros.register_handler("navigation_goal", process_nav)
```

### Web Agent Binding

```python
from src.api import WebAgentBinding

web = WebAgentBinding(api_endpoint="http://localhost:3000/api")
await web.connect()

# Register cognitive endpoint
web.register_handler("cognitive_query", handle_query)
web.register_handler("dashboard_update", send_update)
```

---

## Security & Authentication

### JWT Authentication

```python
from src.api import AuthenticationManager, UserRole, Permission

auth = AuthenticationManager(
    jwt_secret="your-secret-key",
    token_expire_hours=24
)

# Create user
user = auth.create_user(
    username="developer",
    email="dev@example.com",
    role=UserRole.DEVELOPER
)

# Generate token
token = auth.generate_jwt_token(user.user_id, {
    "ip_address": "127.0.0.1"
})

# Verify token
payload = auth.verify_jwt_token(token)
```

### API Keys

```python
# Create API key
raw_key, api_key = auth.create_api_key(
    user_id=user.user_id,
    expires_in_days=30
)

# Verify
verified = auth.verify_api_key(raw_key)
```

### Role-Based Access Control

```python
# Roles and permissions
UserRole.ADMIN      # Full access
UserRole.DEVELOPER  # Read/write/execute
UserRole.ANALYST    # Read/query
UserRole.VIEWER     # Read only
UserRole.SERVICE    # Inter-service communication

# Check permission
has_access = auth.check_permission(user_id, Permission.EXECUTE_TASK)
```

### Rate Limiting

```python
# Check rate limit
is_allowed, info = auth.check_rate_limit(
    resource="query",
    user_id=user.user_id
)

if not is_allowed:
    print(f"Rate limited. Retry after {info['retry_after']}s")
```

---

## Performance Guidelines

### Target Metrics

| Metric | Target | Measured |
|--------|--------|----------|
| API Response Time | < 100ms | ~15-50ms |
| WebSocket Latency | < 50ms | ~10-20ms |
| State Sync | < 100ms | ~25-75ms |
| Concurrent Connections | 1000+ | Tested |
| Task Throughput | 1000/s | ~850/s |

### Optimization Tips

1. **Use async operations** - All I/O should be async
2. **Batch state updates** - Use `submit_task_batch()` for multiple tasks
3. **Subscribe selectively** - Only subscribe to needed event types
4. **Use appropriate query types** - Financial queries are optimized for financial data
5. **Enable circuit breakers** - Prevent cascading failures
6. **Monitor health** - Use the health monitoring system

### Connection Pooling

```python
# For high-load scenarios
api = CognitiveMeshAPI(
    node_id="high_load_node",
    port=8000
)
# Configure connection pool via uvicorn
```

---

## Examples

### Complete Integration Example

```python
import asyncio
from src.api import (
    CognitiveMeshAPI,
    TaskOrchestrator,
    TaskDefinition,
    TaskPriority,
    SelfHealingSystem
)

async def main():
    # 1. Initialize API
    api = CognitiveMeshAPI("master_node", port=8000)
    await api.initialize()
    
    # 2. Setup task orchestration
    orchestrator = TaskOrchestrator("master_node")
    await orchestrator.start()
    
    # 3. Enable self-healing
    healing = SelfHealingSystem("master_node")
    await healing.start()
    
    # 4. Register analysis handler
    async def analyze_data(payload):
        data = payload.get("data", [])
        return {
            "count": len(data),
            "sum": sum(data),
            "avg": sum(data) / len(data) if data else 0
        }
    
    orchestrator.register_handler("analysis", analyze_data)
    
    # 5. Submit and process tasks
    task = TaskDefinition(
        task_id="analysis_001",
        task_type="analysis",
        payload={"data": [1, 2, 3, 4, 5]},
        priority=TaskPriority.HIGH
    )
    
    task_id = await orchestrator.submit_task(task)
    result = await orchestrator.wait_for_task(task_id)
    
    print(f"Result: {result.result_data}")
    
    # 6. Cleanup
    await healing.stop()
    await orchestrator.stop()
    await api.shutdown()

asyncio.run(main())
```

### Real-time Dashboard Integration

```python
from src.api import CognitiveMeshAPI
import websockets

async def dashboard_client():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        # Subscribe to all state events
        await ws.send(json.dumps({
            "type": "subscribe",
            "event_types": ["state_events", "task_updates", "health_alerts"]
        }))
        
        # Listen for events
        async for message in ws:
            event = json.loads(message)
            update_dashboard(event)
```

---

## API Versioning

The API supports versioning for backward compatibility:

- Current: `v1.1.0`
- Supported: `v1.0`, `v1.0.0`, `v1.1.0`
- Minimum: `v1.0`

All endpoints include version prefix: `/api/v1/...`

Version headers are included in responses:
```
X-API-Version: 1.1.0
```

---

## Error Handling

All errors follow a consistent format:

```json
{
    "success": false,
    "message": "Detailed error message",
    "error_code": "RATE_LIMITED",
    "details": {
        "retry_after": 30
    },
    "timestamp": "2025-01-15T10:30:00Z"
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `RATE_LIMITED` | Too many requests |
| `UNAUTHORIZED` | Invalid or missing token |
| `FORBIDDEN` | Insufficient permissions |
| `NOT_FOUND` | Entity not found |
| `TIMEOUT` | Operation timed out |
| `CIRCUIT_OPEN` | Circuit breaker is open |
| `NODE_UNHEALTHY` | Target node is unhealthy |

---

## Support

For issues and contributions, please refer to the repository documentation.
