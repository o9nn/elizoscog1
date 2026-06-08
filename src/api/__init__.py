"""
Distributed Cognitive Mesh APIs - Phase 4 Implementation

This package provides the core API infrastructure for distributed cognitive operations,
including REST/WebSocket endpoints, state synchronization, external system bindings,
task orchestration, and self-healing capabilities.
"""

from .mesh_api import CognitiveMeshAPI
from .websocket_handler import WebSocketHandler
from .state_manager import DistributedStateManager
from .auth_manager import AuthenticationManager
from .external_bindings import UnityBinding, ROSBinding, WebAgentBinding
from .task_orchestrator import (
    TaskOrchestrator, 
    TaskDefinition,
    TaskResult,
    TaskStatus,
    TaskPriority,
    DistributedTaskCoordinator
)
from .self_healing import (
    SelfHealingSystem,
    HealthMonitor,
    LoadBalancer,
    CircuitBreaker,
    HealthStatus,
    CircuitState
)

__all__ = [
    # Core API
    'CognitiveMeshAPI',
    'WebSocketHandler', 
    'DistributedStateManager',
    'AuthenticationManager',
    # External bindings
    'UnityBinding',
    'ROSBinding', 
    'WebAgentBinding',
    # Task orchestration
    'TaskOrchestrator',
    'TaskDefinition',
    'TaskResult',
    'TaskStatus',
    'TaskPriority',
    'DistributedTaskCoordinator',
    # Self-healing
    'SelfHealingSystem',
    'HealthMonitor',
    'LoadBalancer',
    'CircuitBreaker',
    'HealthStatus',
    'CircuitState'
]

__version__ = "1.1.0"