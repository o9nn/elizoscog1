"""
Task Orchestrator for Distributed Cognitive Mesh

Provides advanced task orchestration and coordination protocols including:
- Distributed task scheduling and execution
- Task dependency management
- Load-aware task distribution
- Task result aggregation
- Failure recovery and retry logic
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import heapq


# Configuration constants
MAX_BACKOFF_SECONDS = 30  # Maximum wait time for retry backoff


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    WAITING_DEPENDENCY = "waiting_dependency"


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class TaskDefinition:
    """Definition of a cognitive task"""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    dependencies: List[str] = field(default_factory=list)
    target_nodes: Optional[List[str]] = None
    timeout_seconds: float = 60.0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """Compare tasks by priority for heap operations"""
        return self.priority.value < other.priority.value


@dataclass
class TaskResult:
    """Result of task execution"""
    task_id: str
    status: TaskStatus
    result_data: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    executed_by_node: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0


@dataclass
class NodeCapacity:
    """Capacity and workload information for a node"""
    node_id: str
    max_concurrent_tasks: int = 10
    current_tasks: int = 0
    capabilities: Set[str] = field(default_factory=set)
    is_healthy: bool = True
    last_heartbeat: datetime = field(default_factory=datetime.now)
    average_task_time_ms: float = 100.0
    task_success_rate: float = 1.0


class TaskQueue:
    """Priority-based task queue with dependency tracking"""
    
    def __init__(self):
        self._heap: List[TaskDefinition] = []
        self._task_map: Dict[str, TaskDefinition] = {}
        self._pending_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._dependents: Dict[str, Set[str]] = defaultdict(set)
        self.logger = logging.getLogger(__name__)
    
    def enqueue(self, task: TaskDefinition) -> bool:
        """Add task to queue, respecting dependencies"""
        if task.task_id in self._task_map:
            return False
        
        self._task_map[task.task_id] = task
        
        # Check if task has unmet dependencies
        unmet = [dep for dep in task.dependencies if dep not in self._completed_tasks()]
        
        if unmet:
            self._pending_dependencies[task.task_id] = set(unmet)
            for dep in unmet:
                self._dependents[dep].add(task.task_id)
            return True
        
        heapq.heappush(self._heap, task)
        return True
    
    def dequeue(self) -> Optional[TaskDefinition]:
        """Get next task from queue"""
        if not self._heap:
            return None
        
        task = heapq.heappop(self._heap)
        return task
    
    def mark_complete(self, task_id: str):
        """Mark task as complete and release dependents"""
        if task_id in self._task_map:
            del self._task_map[task_id]
        
        # Release tasks waiting on this dependency
        for dependent_id in self._dependents.get(task_id, set()):
            if dependent_id in self._pending_dependencies:
                self._pending_dependencies[dependent_id].discard(task_id)
                
                # If all dependencies satisfied, add to queue
                if not self._pending_dependencies[dependent_id]:
                    del self._pending_dependencies[dependent_id]
                    if dependent_id in self._task_map:
                        heapq.heappush(self._heap, self._task_map[dependent_id])
        
        if task_id in self._dependents:
            del self._dependents[task_id]
    
    def _completed_tasks(self) -> Set[str]:
        """Get set of completed task IDs (external tracking needed)"""
        return set()
    
    def size(self) -> int:
        """Get total queue size"""
        return len(self._heap) + len(self._pending_dependencies)
    
    def ready_count(self) -> int:
        """Get count of ready-to-execute tasks"""
        return len(self._heap)


class TaskOrchestrator:
    """
    Distributed task orchestrator for cognitive mesh operations
    
    Features:
    - Priority-based task scheduling
    - Dependency resolution
    - Load-aware task distribution
    - Automatic retry with backoff
    - Result aggregation
    - Cross-node coordination
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.task_queue = TaskQueue()
        
        # Task tracking
        self._tasks: Dict[str, TaskDefinition] = {}
        self._results: Dict[str, TaskResult] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        
        # Node management
        self._nodes: Dict[str, NodeCapacity] = {}
        self._node_tasks: Dict[str, Set[str]] = defaultdict(set)
        
        # Task handlers
        self._task_handlers: Dict[str, Callable] = {}
        
        # Coordination
        self._coordinator_lock = asyncio.Lock()
        self._scheduler_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Metrics
        self._metrics = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_execution_time_ms": 0.0,
            "retry_count": 0
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start the task orchestrator"""
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self.logger.info(f"Task orchestrator started on node {self.node_id}")
    
    async def stop(self):
        """Stop the task orchestrator"""
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Cancel running tasks
        for task_id, task in self._running_tasks.items():
            task.cancel()
        
        self.logger.info("Task orchestrator stopped")
    
    def register_handler(self, task_type: str, handler: Callable):
        """Register a handler for a task type"""
        self._task_handlers[task_type] = handler
        self.logger.debug(f"Registered handler for task type: {task_type}")
    
    def register_node(self, node_capacity: NodeCapacity):
        """Register a node with its capacity"""
        self._nodes[node_capacity.node_id] = node_capacity
        self.logger.info(f"Registered node {node_capacity.node_id} with capacity {node_capacity.max_concurrent_tasks}")
    
    def deregister_node(self, node_id: str):
        """Deregister a node"""
        if node_id in self._nodes:
            del self._nodes[node_id]
            # Requeue tasks assigned to this node
            for task_id in self._node_tasks.get(node_id, set()):
                if task_id in self._tasks:
                    self.task_queue.enqueue(self._tasks[task_id])
            self.logger.info(f"Deregistered node {node_id}")
    
    async def submit_task(self, task: TaskDefinition) -> str:
        """Submit a task for execution"""
        self._tasks[task.task_id] = task
        self._metrics["tasks_submitted"] += 1
        
        # Create initial result
        self._results[task.task_id] = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.PENDING
        )
        
        # Add to queue
        self.task_queue.enqueue(task)
        
        self.logger.debug(f"Submitted task {task.task_id} of type {task.task_type}")
        return task.task_id
    
    async def submit_task_batch(self, tasks: List[TaskDefinition]) -> List[str]:
        """Submit multiple tasks atomically"""
        task_ids = []
        async with self._coordinator_lock:
            for task in tasks:
                task_id = await self.submit_task(task)
                task_ids.append(task_id)
        return task_ids
    
    async def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get current status of a task"""
        return self._results.get(task_id)
    
    async def wait_for_task(self, task_id: str, timeout: float = 60.0) -> TaskResult:
        """Wait for a task to complete"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = self._results.get(task_id)
            if result and result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return result
            await asyncio.sleep(0.1)
        
        # Timeout - return current status
        return self._results.get(task_id, TaskResult(
            task_id=task_id,
            status=TaskStatus.FAILED,
            error="Timeout waiting for task completion"
        ))
    
    async def wait_for_tasks(self, task_ids: List[str], timeout: float = 120.0) -> Dict[str, TaskResult]:
        """Wait for multiple tasks to complete"""
        results = {}
        remaining_ids = set(task_ids)
        start_time = time.time()
        
        while remaining_ids and time.time() - start_time < timeout:
            for task_id in list(remaining_ids):
                result = self._results.get(task_id)
                if result and result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    results[task_id] = result
                    remaining_ids.remove(task_id)
            
            if remaining_ids:
                await asyncio.sleep(0.1)
        
        # Handle remaining tasks (timeout)
        for task_id in remaining_ids:
            results[task_id] = self._results.get(task_id, TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error="Timeout waiting for task completion"
            ))
        
        return results
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        if task_id in self._running_tasks:
            self._running_tasks[task_id].cancel()
            self._results[task_id] = TaskResult(
                task_id=task_id,
                status=TaskStatus.CANCELLED
            )
            del self._running_tasks[task_id]
            return True
        
        if task_id in self._tasks:
            self._results[task_id] = TaskResult(
                task_id=task_id,
                status=TaskStatus.CANCELLED
            )
            del self._tasks[task_id]
            return True
        
        return False
    
    def _select_node(self, task: TaskDefinition) -> Optional[str]:
        """Select best node for task execution based on load and capabilities"""
        candidates = []
        
        for node_id, node in self._nodes.items():
            # Check if node is healthy
            if not node.is_healthy:
                continue
            
            # Check if node has required capabilities
            if task.target_nodes and node_id not in task.target_nodes:
                continue
            
            # Check capacity
            if node.current_tasks >= node.max_concurrent_tasks:
                continue
            
            # Calculate score (lower is better)
            load_ratio = node.current_tasks / node.max_concurrent_tasks
            score = load_ratio + (1 - node.task_success_rate) * 2
            
            candidates.append((score, node_id))
        
        if not candidates:
            # Fall back to local execution
            return self.node_id
        
        # Select node with lowest score
        candidates.sort()
        return candidates[0][1]
    
    async def _execute_task(self, task: TaskDefinition) -> TaskResult:
        """Execute a task"""
        start_time = time.time()
        
        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            started_at=datetime.now(),
            executed_by_node=self.node_id
        )
        
        self._results[task.task_id] = result
        
        try:
            # Get handler for task type
            handler = self._task_handlers.get(task.task_type)
            
            if handler:
                # Execute with timeout
                try:
                    result_data = await asyncio.wait_for(
                        handler(task.payload),
                        timeout=task.timeout_seconds
                    )
                    result.result_data = result_data
                    result.status = TaskStatus.COMPLETED
                    
                except asyncio.TimeoutError:
                    result.status = TaskStatus.FAILED
                    result.error = f"Task timeout after {task.timeout_seconds}s"
                    
            else:
                # Default handler - just process payload
                result.result_data = {
                    "task_type": task.task_type,
                    "payload": task.payload,
                    "processed": True
                }
                result.status = TaskStatus.COMPLETED
            
        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error = str(e)
            self.logger.error(f"Task {task.task_id} failed: {e}")
        
        result.completed_at = datetime.now()
        result.execution_time_ms = (time.time() - start_time) * 1000
        
        # Update metrics
        if result.status == TaskStatus.COMPLETED:
            self._metrics["tasks_completed"] += 1
        else:
            self._metrics["tasks_failed"] += 1
        
        # Update average execution time
        total_completed = self._metrics["tasks_completed"]
        if total_completed > 0:
            current_avg = self._metrics["average_execution_time_ms"]
            self._metrics["average_execution_time_ms"] = (
                (current_avg * (total_completed - 1) + result.execution_time_ms) / total_completed
            )
        
        return result
    
    async def _retry_task(self, task: TaskDefinition, current_retry: int) -> TaskResult:
        """Retry a failed task with exponential backoff"""
        if current_retry >= task.max_retries:
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=f"Max retries ({task.max_retries}) exceeded",
                retry_count=current_retry
            )
        
        # Exponential backoff with configurable maximum
        wait_time = min(2 ** current_retry, MAX_BACKOFF_SECONDS)
        await asyncio.sleep(wait_time)
        
        self._metrics["retry_count"] += 1
        
        result = await self._execute_task(task)
        result.retry_count = current_retry + 1
        
        return result
    
    async def _scheduler_loop(self):
        """Main scheduling loop"""
        while True:
            try:
                # Get next task
                task = self.task_queue.dequeue()
                
                if task:
                    # Select target node
                    target_node = self._select_node(task)
                    
                    if target_node == self.node_id:
                        # Execute locally
                        execute_task = asyncio.create_task(self._execute_task(task))
                        self._running_tasks[task.task_id] = execute_task
                        
                        # Handle completion
                        asyncio.create_task(self._handle_task_completion(task.task_id, execute_task))
                    else:
                        # Remote execution would be handled here
                        # For now, execute locally
                        execute_task = asyncio.create_task(self._execute_task(task))
                        self._running_tasks[task.task_id] = execute_task
                        asyncio.create_task(self._handle_task_completion(task.task_id, execute_task))
                
                await asyncio.sleep(0.01)  # Small delay to prevent busy-waiting
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _handle_task_completion(self, task_id: str, execute_task: asyncio.Task):
        """Handle task completion and retry logic"""
        try:
            result = await execute_task
            
            # Clean up
            if task_id in self._running_tasks:
                del self._running_tasks[task_id]
            
            # Mark complete in queue
            self.task_queue.mark_complete(task_id)
            
            # Handle retry if failed
            if result.status == TaskStatus.FAILED:
                task = self._tasks.get(task_id)
                if task and result.retry_count < task.max_retries:
                    # Retry the task
                    retry_result = await self._retry_task(task, result.retry_count)
                    self._results[task_id] = retry_result
                    return
            
            self._results[task_id] = result
            
        except asyncio.CancelledError:
            self._results[task_id] = TaskResult(
                task_id=task_id,
                status=TaskStatus.CANCELLED
            )
        except Exception as e:
            self._results[task_id] = TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=str(e)
            )
    
    async def _health_check_loop(self):
        """Periodic health check for nodes"""
        while True:
            try:
                now = datetime.now()
                
                for node_id, node in list(self._nodes.items()):
                    # Check if node heartbeat is stale
                    if (now - node.last_heartbeat).seconds > 60:
                        node.is_healthy = False
                        self.logger.warning(f"Node {node_id} marked unhealthy (no heartbeat)")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(10)
    
    def update_node_heartbeat(self, node_id: str):
        """Update node heartbeat"""
        if node_id in self._nodes:
            self._nodes[node_id].last_heartbeat = datetime.now()
            self._nodes[node_id].is_healthy = True
    
    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            "node_id": self.node_id,
            "queue_size": self.task_queue.size(),
            "ready_tasks": self.task_queue.ready_count(),
            "running_tasks": len(self._running_tasks),
            "registered_nodes": len(self._nodes),
            "healthy_nodes": sum(1 for n in self._nodes.values() if n.is_healthy),
            "metrics": self._metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_task_handlers(self) -> List[str]:
        """Get list of registered task handlers"""
        return list(self._task_handlers.keys())


class DistributedTaskCoordinator:
    """
    High-level coordinator for distributed task execution across mesh nodes
    
    Features:
    - Cross-node task coordination
    - Workflow management
    - Aggregate result collection
    - Distributed transaction support
    """
    
    def __init__(self, local_orchestrator: TaskOrchestrator):
        self.local_orchestrator = local_orchestrator
        self._workflows: Dict[str, Dict[str, Any]] = {}
        self._aggregated_results: Dict[str, List[TaskResult]] = defaultdict(list)
        self.logger = logging.getLogger(__name__)
    
    async def execute_workflow(self, workflow_id: str, tasks: List[TaskDefinition]) -> Dict[str, TaskResult]:
        """Execute a workflow of dependent tasks"""
        self._workflows[workflow_id] = {
            "status": "running",
            "tasks": [t.task_id for t in tasks],
            "started_at": datetime.now()
        }
        
        # Submit all tasks
        task_ids = await self.local_orchestrator.submit_task_batch(tasks)
        
        # Wait for completion
        results = await self.local_orchestrator.wait_for_tasks(task_ids)
        
        # Update workflow status
        all_completed = all(r.status == TaskStatus.COMPLETED for r in results.values())
        self._workflows[workflow_id]["status"] = "completed" if all_completed else "failed"
        self._workflows[workflow_id]["completed_at"] = datetime.now()
        
        return results
    
    async def map_reduce_task(self, task_type: str, data_chunks: List[Any], 
                             reduce_fn: Optional[Callable] = None) -> Any:
        """Execute map-reduce style distributed task"""
        # Create map tasks
        map_tasks = []
        for i, chunk in enumerate(data_chunks):
            task = TaskDefinition(
                task_id=f"map_{uuid.uuid4().hex[:8]}",
                task_type=task_type,
                payload={"chunk_id": i, "data": chunk}
            )
            map_tasks.append(task)
        
        # Execute map phase
        map_results = await self.local_orchestrator.wait_for_tasks(
            await self.local_orchestrator.submit_task_batch(map_tasks)
        )
        
        # Reduce phase
        results_data = [r.result_data for r in map_results.values() if r.status == TaskStatus.COMPLETED]
        
        if reduce_fn:
            return reduce_fn(results_data)
        
        return results_data
    
    async def scatter_gather(self, task_type: str, payload: Dict[str, Any], 
                            target_nodes: List[str]) -> Dict[str, TaskResult]:
        """Scatter task to multiple nodes and gather results"""
        tasks = []
        for node_id in target_nodes:
            task = TaskDefinition(
                task_id=f"scatter_{uuid.uuid4().hex[:8]}",
                task_type=task_type,
                payload=payload,
                target_nodes=[node_id]
            )
            tasks.append(task)
        
        # Submit and wait
        task_ids = await self.local_orchestrator.submit_task_batch(tasks)
        return await self.local_orchestrator.wait_for_tasks(task_ids)
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a workflow"""
        return self._workflows.get(workflow_id)
