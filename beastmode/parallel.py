#!/usr/bin/env python3
"""
BEASTMODE Parallel Execution & Async Pipeline
==============================================

Advanced parallelism primitives for the inference engine.

Features:
- Work-stealing scheduler with per-worker deques and dependency graphs
- NUMA-aware worker count derivation from hardware topology
- Bucketed batching for similar-shaped tensors with latency SLA
- Asynchronous operation pipeline with double-buffering and callbacks
"""

import asyncio
import logging
import time
import threading
from collections import deque, defaultdict
from typing import (
    Dict, List, Any, Optional, Tuple, Callable, Hashable, Set
)
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .hardware import detect_cpu_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 3.1 — Work-Stealing Scheduler
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A unit of schedulable work with optional dependencies"""
    task_id: Hashable
    fn: Callable[..., Any]
    args: Tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    depends_on: Set[Hashable] = field(default_factory=set)
    priority: int = 0  # Higher = more urgent
    result: Any = None
    error: Optional[Exception] = None
    completed: bool = False


class DependencyGraph:
    """
    Directed acyclic graph (DAG) for task dependency resolution.

    Tracks which tasks are ready to run (all dependencies satisfied)
    and which are still blocked.
    """

    def __init__(self):
        self._tasks: Dict[Hashable, Task] = {}
        self._dependents: Dict[Hashable, Set[Hashable]] = defaultdict(set)
        self._in_degree: Dict[Hashable, int] = {}

    def add_task(self, task: Task) -> None:
        """Add a task with its dependencies to the graph"""
        self._tasks[task.task_id] = task
        self._in_degree[task.task_id] = len(task.depends_on)
        for dep in task.depends_on:
            self._dependents[dep].add(task.task_id)

    def get_ready(self) -> List[Task]:
        """Return all tasks whose dependencies are fully satisfied"""
        ready = [
            self._tasks[tid] for tid, deg in self._in_degree.items()
            if deg == 0 and not self._tasks[tid].completed
            and self._tasks[tid].error is None
        ]
        # Sort by priority descending for urgency-aware scheduling
        ready.sort(key=lambda t: -t.priority)
        return ready

    def mark_completed(self, task_id: Hashable) -> List[Task]:
        """
        Mark a task as completed and return newly-unblocked tasks.
        """
        task = self._tasks[task_id]
        task.completed = True
        newly_ready = []
        for dependent_id in self._dependents.get(task_id, set()):
            self._in_degree[dependent_id] -= 1
            if self._in_degree[dependent_id] == 0:
                newly_ready.append(self._tasks[dependent_id])
        return newly_ready

    def mark_failed(self, task_id: Hashable, error: Exception) -> None:
        """Mark a task as failed"""
        self._tasks[task_id].error = error

    @property
    def all_done(self) -> bool:
        """True when every task is completed or failed"""
        return all(
            t.completed or t.error is not None for t in self._tasks.values()
        )

    @property
    def pending_count(self) -> int:
        return sum(
            1 for t in self._tasks.values()
            if not t.completed and t.error is None
        )


class WorkStealingScheduler:
    """
    Work-stealing task scheduler for load-balanced parallel execution.

    Each worker owns a deque of tasks. Workers push/pop from their own
    deque (LIFO for cache locality) and steal from the tail of other
    workers' deques (FIFO) when idle. This minimizes contention and
    keeps data hot in per-core caches.

    NUMA-aware: worker count defaults to physical core count when
    NUMA topology is detected, avoiding cross-socket memory stalls.
    """

    def __init__(self, num_workers: Optional[int] = None):
        if num_workers is not None:
            self.num_workers = num_workers
        else:
            cpu = detect_cpu_features()
            # Prefer physical cores on multi-socket/NUMA systems to
            # avoid cross-socket memory traffic on shared queues.
            if cpu.numa_nodes > 1 and cpu.physical_cores:
                self.num_workers = cpu.physical_cores
            else:
                self.num_workers = cpu.cpu_count

        # Per-worker task queues (index = worker id)
        self._queues: List[deque] = [deque() for _ in range(self.num_workers)]
        self._locks = [threading.Lock() for _ in range(self.num_workers)]

        # Round-robin seed for initial task assignment
        self._next_worker = 0
        self._assignment_lock = threading.Lock()

        # Statistics
        self.tasks_submitted = 0
        self.tasks_completed = 0
        self.steal_count = 0

        logger.info(
            f"WorkStealingScheduler initialized: workers={self.num_workers}"
        )

    def submit(self, task: Task, worker_hint: Optional[int] = None) -> int:
        """
        Submit a task to a worker queue. Returns the worker index.
        """
        if worker_hint is not None:
            worker = worker_hint % self.num_workers
        else:
            with self._assignment_lock:
                worker = self._next_worker
                self._next_worker = (self._next_worker + 1) % self.num_workers

        with self._locks[worker]:
            self._queues[worker].append(task)  # push to tail
        self.tasks_submitted += 1
        return worker

    def _pop_own(self, worker: int) -> Optional[Task]:
        """Pop from own queue (LIFO for locality)"""
        with self._locks[worker]:
            if self._queues[worker]:
                return self._queues[worker].pop()  # pop from tail (LIFO)
        return None

    def _steal(self, thief: int) -> Optional[Task]:
        """Steal from another worker's queue (FIFO from head)"""
        for offset in range(1, self.num_workers):
            victim = (thief + offset) % self.num_workers
            with self._locks[victim]:
                if self._queues[victim]:
                    self.steal_count += 1
                    return self._queues[victim].popleft()  # steal from head
        return None

    def run_all(self, tasks: List[Task]) -> Dict[Hashable, Any]:
        """
        Execute a set of tasks with dependency resolution and work stealing.

        Tasks are scheduled respecting dependencies; workers steal from
        each other when idle for automatic load balancing.

        Returns:
            Dict mapping task_id to result (or exception for failures).
        """
        if not tasks:
            return {}

        graph = DependencyGraph()
        for task in tasks:
            graph.add_task(task)

        results: Dict[Hashable, Any] = {}
        results_lock = threading.Lock()
        enqueued: Set[Hashable] = set()
        enqueue_lock = threading.Lock()
        done_event = threading.Event()

        def _try_enqueue(task: Task) -> bool:
            """Atomically try to enqueue a task (returns True if we won)"""
            with enqueue_lock:
                if task.task_id in enqueued:
                    return False
                enqueued.add(task.task_id)
            self.submit(task)
            return True

        def _worker_loop(worker_id: int):
            while not done_event.is_set():
                task = self._pop_own(worker_id)
                if task is None:
                    task = self._steal(worker_id)
                if task is None:
                    # Check if there are pending ready tasks not yet enqueued
                    if graph.pending_count == 0:
                        break
                    # Try to enqueue any ready-but-not-yet-submitted tasks
                    for ready in graph.get_ready():
                        _try_enqueue(ready)
                    # Brief yield to avoid busy-spinning
                    time.sleep(0.0001)
                    continue

                # Execute the task
                try:
                    task.result = task.fn(*task.args, **task.kwargs)
                    with results_lock:
                        results[task.task_id] = task.result
                    newly_ready = graph.mark_completed(task.task_id)
                except Exception as e:
                    task.error = e
                    with results_lock:
                        results[task.task_id] = e
                    graph.mark_failed(task_id=task.task_id, error=e)
                    newly_ready = []

                self.tasks_completed += 1

                # Feed newly unblocked tasks back into queues
                for new_task in newly_ready:
                    _try_enqueue(new_task)

        # Submit initial ready tasks
        for task in graph.get_ready():
            _try_enqueue(task)

        threads = []
        for wid in range(self.num_workers):
            t = threading.Thread(target=_worker_loop, args=(wid,), daemon=True)
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30.0)
            if t.is_alive():
                logger.warning(
                    f"Worker thread did not finish within 30s timeout; "
                    f"results may be incomplete "
                    f"({len(results)}/{len(tasks)} tasks done)"
                )

        return results

    def execute_sync(self, fn: Callable, *args, **kwargs) -> Any:
        """Execute a single function synchronously (convenience)"""
        task = Task(task_id='sync', fn=fn, args=args, kwargs=kwargs)
        results = self.run_all([task])
        result = results.get('sync')
        if isinstance(result, Exception):
            raise result
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        queue_depths = [len(q) for q in self._queues]
        return {
            'num_workers': self.num_workers,
            'tasks_submitted': self.tasks_submitted,
            'tasks_completed': self.tasks_completed,
            'steal_count': self.steal_count,
            'queue_depths': queue_depths,
            'steal_ratio': self.steal_count / max(self.tasks_completed, 1),
        }


# ---------------------------------------------------------------------------
# Phase 3.2 — Bucketed Dynamic Batching
# ---------------------------------------------------------------------------

@dataclass
class BatchSLA:
    """Latency service-level agreement for batch processing"""
    max_latency_ms: float = 5.0
    max_batch_size: int = 256
    min_batch_size: int = 1


class BucketedBatcher:
    """
    Groups similar-shaped tensors into buckets for efficient batching.

    Eliminates padding waste by only batching tensors with identical
    (or compatible) shapes. Adapts batch sizes to meet a latency SLA
    via a feedback loop on observed per-batch latency.
    """

    def __init__(self, sla: Optional[BatchSLA] = None):
        self.sla = sla or BatchSLA()
        # Buckets keyed by shape tuple
        self._buckets: Dict[Tuple[int, ...], List[np.ndarray]] = defaultdict(list)
        # Adaptive batch size per bucket
        self._batch_sizes: Dict[Tuple[int, ...], int] = {}
        # Latency history per bucket for SLA feedback
        self._latency_history: Dict[Tuple[int, ...], List[float]] = defaultdict(list)

        self.total_batches = 0
        self.total_items = 0

        logger.info(
            f"BucketedBatcher initialized: SLA={self.sla.max_latency_ms}ms"
        )

    def add(self, tensor: np.ndarray) -> None:
        """Add a tensor to the appropriate shape bucket"""
        self._buckets[tensor.shape].append(tensor)

    def get_ready_batches(self) -> List[Tuple[Tuple[int, ...], List[np.ndarray]]]:
        """
        Extract all full batches ready for processing.

        Returns list of (shape, batch_tensors) where each batch
        is sized according to the adaptive per-bucket batch size.
        """
        ready = []
        for shape, tensors in self._buckets.items():
            target_size = self._batch_sizes.get(shape, self.sla.min_batch_size)
            while len(tensors) >= target_size:
                batch = tensors[:target_size]
                del tensors[:target_size]
                ready.append((shape, batch))
                self.total_batches += 1
                self.total_items += len(batch)
        return ready

    def flush_all(self) -> List[Tuple[Tuple[int, ...], List[np.ndarray]]]:
        """
        Flush all buckets regardless of batch size. Returns remaining items.
        """
        flushed = []
        for shape, tensors in self._buckets.items():
            if tensors:
                flushed.append((shape, list(tensors)))
                self.total_batches += 1
                self.total_items += len(tensors)
                tensors.clear()
        return flushed

    def record_latency(self, shape: Tuple[int, ...],
                       batch_size: int, latency_ms: float) -> None:
        """
        Record observed latency for a batch and adapt batch size.

        If latency exceeds SLA, shrink batch size. If well under SLA,
        grow it (multiplicative increase / multiplicative decrease).
        """
        self._latency_history[shape].append(latency_ms)
        current = self._batch_sizes.get(shape, self.sla.min_batch_size)

        if latency_ms > self.sla.max_latency_ms:
            # SLA violated: halve batch size
            new_size = max(current // 2, self.sla.min_batch_size)
        elif latency_ms < self.sla.max_latency_ms * 0.5:
            # Well under SLA: grow batch
            new_size = min(current * 2, self.sla.max_batch_size)
        else:
            new_size = current

        self._batch_sizes[shape] = new_size

    @property
    def pending_count(self) -> int:
        return sum(len(v) for v in self._buckets.values())

    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_batches': self.total_batches,
            'total_items': self.total_items,
            'pending_items': self.pending_count,
            'bucket_count': len(self._buckets),
            'batch_sizes': {
                str(k): v for k, v in self._batch_sizes.items()
            },
        }


# ---------------------------------------------------------------------------
# Phase 3.3 — Asynchronous Operation Pipeline
# ---------------------------------------------------------------------------

class PipelineStage(Enum):
    """Stages in the async operation pipeline"""
    PENDING = "pending"
    COMPUTING = "computing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class PipelineSlot:
    """One slot in the double-buffer pipeline"""
    slot_id: int
    data: Optional[np.ndarray] = None
    result: Optional[np.ndarray] = None
    stage: PipelineStage = PipelineStage.PENDING
    callback: Optional[Callable] = None
    submitted_at: float = 0.0
    completed_at: float = 0.0

    @property
    def latency_ms(self) -> float:
        if self.completed_at > 0 and self.submitted_at > 0:
            return (self.completed_at - self.submitted_at) * 1000
        return 0.0


class AsyncPipeline:
    """
    Double-buffered asynchronous operation pipeline.

    Overlaps compute with data movement: while one buffer is being
    computed on, the other is being filled with the next input. Async
    callbacks enable non-blocking result delivery.

    Usage:
        pipeline = AsyncPipeline(compute_fn=my_kernel)
        await pipeline.start()
        pipeline.submit(data, callback=on_result)
        await pipeline.drain()
    """

    def __init__(self, compute_fn: Optional[Callable] = None,
                 buffer_count: int = 2):
        self.compute_fn = compute_fn or (lambda x: x)
        self.buffer_count = max(buffer_count, 2)

        self._slots: List[PipelineSlot] = [
            PipelineSlot(slot_id=i) for i in range(self.buffer_count)
        ]
        self._pending: deque = deque()
        self._lock = asyncio.Lock()
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None

        # Stats
        self.total_submitted = 0
        self.total_completed = 0
        self.total_failed = 0
        self.latencies: List[float] = []

        logger.info(
            f"AsyncPipeline initialized: buffers={self.buffer_count}"
        )

    async def start(self) -> None:
        """Start the pipeline worker"""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        """Stop the pipeline worker"""
        self._running = False
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    def submit(self, data: np.ndarray,
               callback: Optional[Callable] = None) -> int:
        """
        Submit data for async processing.

        Args:
            data: Input tensor.
            callback: Optional callable invoked with the result.

        Returns:
            Submission index (monotonically increasing).
        """
        idx = self.total_submitted
        self._pending.append((data, callback))
        self.total_submitted += 1
        return idx

    async def drain(self, timeout_s: float = 10.0) -> List[Any]:
        """
        Wait until all submitted items are processed.

        Returns:
            List of results in submission order.
        """
        deadline = time.monotonic() + timeout_s
        while self._pending and time.monotonic() < deadline:
            await asyncio.sleep(0.001)
        # Wait for in-flight slots
        while time.monotonic() < deadline:
            if all(s.stage in (PipelineStage.PENDING, PipelineStage.COMPLETE,
                               PipelineStage.FAILED)
                   for s in self._slots):
                break
            await asyncio.sleep(0.001)

        return [s.result for s in self._slots if s.result is not None]

    async def _worker(self) -> None:
        """Background worker that processes pending items"""
        while self._running:
            if not self._pending:
                await asyncio.sleep(0.0001)
                continue

            try:
                data, callback = self._pending.popleft()
            except IndexError:
                continue

            # Find a free slot (double-buffering)
            slot = None
            for s in self._slots:
                if s.stage in (PipelineStage.PENDING, PipelineStage.COMPLETE,
                               PipelineStage.FAILED):
                    slot = s
                    break

            if slot is None:
                # All slots busy; put back and retry
                self._pending.appendleft((data, callback))
                await asyncio.sleep(0.0001)
                continue

            slot.data = data
            slot.callback = callback
            slot.stage = PipelineStage.COMPUTING
            slot.submitted_at = time.monotonic()
            slot.result = None

            try:
                # Run compute in executor to avoid blocking event loop
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, self.compute_fn, data)
                slot.result = result
                slot.stage = PipelineStage.COMPLETE
                slot.completed_at = time.monotonic()
                self.total_completed += 1
                self.latencies.append(slot.latency_ms)

                if callback is not None:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(result)
                        else:
                            callback(result)
                    except Exception as cb_err:
                        logger.warning(f"Pipeline callback failed: {cb_err}")

            except Exception as e:
                slot.stage = PipelineStage.FAILED
                slot.completed_at = time.monotonic()
                self.total_failed += 1
                logger.error(f"Pipeline compute failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        avg_lat = np.mean(self.latencies) if self.latencies else 0.0
        p99_lat = (
            float(np.percentile(self.latencies, 99))
            if len(self.latencies) > 1 else avg_lat
        )
        return {
            'total_submitted': self.total_submitted,
            'total_completed': self.total_completed,
            'total_failed': self.total_failed,
            'pending': len(self._pending),
            'avg_latency_ms': float(avg_lat),
            'p99_latency_ms': p99_lat,
            'buffer_count': self.buffer_count,
        }


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def create_work_stealing_scheduler(
        num_workers: Optional[int] = None) -> WorkStealingScheduler:
    """Create a work-stealing scheduler with NUMA-aware defaults"""
    return WorkStealingScheduler(num_workers=num_workers)


def create_bucketed_batcher(max_latency_ms: float = 5.0,
                            max_batch_size: int = 256) -> BucketedBatcher:
    """Create a bucketed batcher with the given SLA"""
    return BucketedBatcher(sla=BatchSLA(
        max_latency_ms=max_latency_ms,
        max_batch_size=max_batch_size,
    ))


def create_async_pipeline(compute_fn: Optional[Callable] = None,
                          buffer_count: int = 2) -> AsyncPipeline:
    """Create a double-buffered async pipeline"""
    return AsyncPipeline(compute_fn=compute_fn, buffer_count=buffer_count)
