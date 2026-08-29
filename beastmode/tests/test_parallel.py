#!/usr/bin/env python3
"""
Tests for BEASTMODE Phase 2.2, 3.x, 5.2 components:
- Work-stealing scheduler with dependency graphs
- Bucketed dynamic batching with latency SLA
- Asynchronous operation pipeline with double-buffering
- Persistent kernel profile cache with AOT warmup
- Bayesian hyperparameter tuner
- Workload characterization clustering
- Memory/compute tradeoff optimization
"""

import asyncio
import unittest
import numpy as np
import time
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from beastmode.parallel import (
    Task, DependencyGraph, WorkStealingScheduler,
    BucketedBatcher, BatchSLA,
    AsyncPipeline, PipelineStage,
    create_work_stealing_scheduler, create_bucketed_batcher,
    create_async_pipeline,
)
from beastmode.kernel_cache import (
    KernelProfileCache, SpecializedKernelProfile, classify_shape,
    create_kernel_cache,
)
from beastmode.self_tuning import (
    BayesianTuner, WorkloadClusterer, TradeoffOptimizer,
    create_bayesian_tuner, create_workload_clusterer,
    create_tradeoff_optimizer,
)


# ===========================================================================
# Phase 3.1: Work-Stealing Scheduler
# ===========================================================================

class TestDependencyGraph(unittest.TestCase):
    """Test task dependency graph"""

    def test_no_dependencies_all_ready(self):
        graph = DependencyGraph()
        graph.add_task(Task(task_id='a', fn=lambda: 1))
        graph.add_task(Task(task_id='b', fn=lambda: 2))
        ready = graph.get_ready()
        self.assertEqual(len(ready), 2)

    def test_dependency_blocks_task(self):
        graph = DependencyGraph()
        graph.add_task(Task(task_id='a', fn=lambda: 1))
        graph.add_task(Task(task_id='b', fn=lambda: 2, depends_on={'a'}))
        ready_ids = {t.task_id for t in graph.get_ready()}
        self.assertIn('a', ready_ids)
        self.assertNotIn('b', ready_ids)

    def test_completion_unblocks_dependents(self):
        graph = DependencyGraph()
        graph.add_task(Task(task_id='a', fn=lambda: 1))
        graph.add_task(Task(task_id='b', fn=lambda: 2, depends_on={'a'}))
        newly_ready = graph.mark_completed('a')
        self.assertEqual(len(newly_ready), 1)
        self.assertEqual(newly_ready[0].task_id, 'b')

    def test_chain_dependencies(self):
        graph = DependencyGraph()
        graph.add_task(Task(task_id='a', fn=lambda: 1))
        graph.add_task(Task(task_id='b', fn=lambda: 2, depends_on={'a'}))
        graph.add_task(Task(task_id='c', fn=lambda: 3, depends_on={'b'}))
        self.assertEqual(graph.pending_count, 3)
        graph.mark_completed('a')
        self.assertEqual(graph.pending_count, 2)
        graph.mark_completed('b')
        graph.mark_completed('c')
        self.assertTrue(graph.all_done)

    def test_priority_ordering(self):
        graph = DependencyGraph()
        graph.add_task(Task(task_id='low', fn=lambda: 1, priority=1))
        graph.add_task(Task(task_id='high', fn=lambda: 2, priority=10))
        ready = graph.get_ready()
        self.assertEqual(ready[0].task_id, 'high')


class TestWorkStealingScheduler(unittest.TestCase):
    """Test work-stealing task scheduler"""

    def test_num_workers_default(self):
        scheduler = WorkStealingScheduler()
        self.assertGreaterEqual(scheduler.num_workers, 1)

    def test_num_workers_explicit(self):
        scheduler = WorkStealingScheduler(num_workers=2)
        self.assertEqual(scheduler.num_workers, 2)

    def test_run_independent_tasks(self):
        scheduler = WorkStealingScheduler(num_workers=2)
        tasks = [
            Task(task_id=f't{i}', fn=lambda x=i: x * 2)
            for i in range(10)
        ]
        results = scheduler.run_all(tasks)
        self.assertEqual(len(results), 10)
        for i in range(10):
            self.assertEqual(results[f't{i}'], i * 2)

    def test_run_dependent_tasks(self):
        scheduler = WorkStealingScheduler(num_workers=2)

        def step_a():
            return 10

        def step_b():
            return 20

        tasks = [
            Task(task_id='a', fn=step_a),
            Task(task_id='b', fn=step_b, depends_on={'a'}),
        ]
        results = scheduler.run_all(tasks)
        self.assertEqual(results['a'], 10)
        self.assertEqual(results['b'], 20)

    def test_task_failure_captured(self):
        scheduler = WorkStealingScheduler(num_workers=1)

        def failing():
            raise ValueError("intentional")

        tasks = [Task(task_id='fail', fn=failing)]
        results = scheduler.run_all(tasks)
        self.assertIsInstance(results['fail'], ValueError)

    def test_stats(self):
        scheduler = WorkStealingScheduler(num_workers=2)
        tasks = [Task(task_id=f't{i}', fn=lambda: 42) for i in range(5)]
        scheduler.run_all(tasks)
        stats = scheduler.get_stats()
        self.assertEqual(stats['tasks_completed'], 5)
        self.assertEqual(stats['num_workers'], 2)
        self.assertIn('steal_count', stats)

    def test_factory(self):
        scheduler = create_work_stealing_scheduler(num_workers=3)
        self.assertEqual(scheduler.num_workers, 3)


# ===========================================================================
# Phase 3.2: Bucketed Dynamic Batching
# ===========================================================================

class TestBucketedBatcher(unittest.TestCase):
    """Test bucketed batching with SLA"""

    def setUp(self):
        self.batcher = BucketedBatcher(
            sla=BatchSLA(max_latency_ms=5.0, max_batch_size=8, min_batch_size=1)
        )

    def test_add_and_flush(self):
        self.batcher.add(np.zeros((4, 4)))
        self.batcher.add(np.zeros((4, 4)))
        self.batcher.add(np.zeros((8, 8)))
        flushed = self.batcher.flush_all()
        self.assertEqual(len(flushed), 2)  # Two shape groups

    def test_shape_grouping(self):
        self.batcher.add(np.zeros((4, 4)))
        self.batcher.add(np.zeros((8, 8)))
        self.batcher.add(np.zeros((4, 4)))
        self.assertEqual(self.batcher.pending_count, 3)
        flushed = self.batcher.flush_all()
        shapes = {s for s, _ in flushed}
        self.assertEqual(len(shapes), 2)

    def test_sla_shrinks_batch(self):
        shape = (4, 4)
        self.batcher._batch_sizes[shape] = 8
        self.batcher.record_latency(shape, 8, 10.0)  # Over 5ms SLA
        self.assertLessEqual(self.batcher._batch_sizes[shape], 4)

    def test_sla_grows_batch(self):
        shape = (4, 4)
        self.batcher._batch_sizes[shape] = 2
        self.batcher.record_latency(shape, 2, 1.0)  # Well under SLA
        self.assertEqual(self.batcher._batch_sizes[shape], 4)

    def test_sla_maintains_batch(self):
        shape = (4, 4)
        self.batcher._batch_sizes[shape] = 4
        self.batcher.record_latency(shape, 4, 4.0)  # Within SLA
        self.assertEqual(self.batcher._batch_sizes[shape], 4)

    def test_batch_size_capped(self):
        shape = (4, 4)
        self.batcher._batch_sizes[shape] = 8
        self.batcher.record_latency(shape, 8, 0.5)  # Very fast
        # Should cap at max_batch_size=8
        self.assertLessEqual(self.batcher._batch_sizes[shape], 8)

    def test_stats(self):
        self.batcher.add(np.zeros((2, 2)))
        self.batcher.flush_all()
        stats = self.batcher.get_stats()
        self.assertEqual(stats['total_batches'], 1)
        self.assertEqual(stats['total_items'], 1)

    def test_factory(self):
        b = create_bucketed_batcher(max_latency_ms=10.0)
        self.assertEqual(b.sla.max_latency_ms, 10.0)


# ===========================================================================
# Phase 3.3: Async Pipeline
# ===========================================================================

class TestAsyncPipeline(unittest.TestCase):
    """Test async operation pipeline with double-buffering"""

    def test_submit_and_drain(self):
        async def run():
            pipeline = AsyncPipeline(compute_fn=lambda x: x * 2)
            await pipeline.start()
            pipeline.submit(np.array([1.0, 2.0]))
            pipeline.submit(np.array([3.0, 4.0]))
            results = await pipeline.drain(timeout_s=5.0)
            await pipeline.stop()
            return pipeline, results

        pipeline, results = asyncio.run(run())
        self.assertGreaterEqual(pipeline.total_completed, 1)

    def test_callback_invoked(self):
        collected = []

        async def run():
            pipeline = AsyncPipeline(compute_fn=lambda x: x + 1)
            await pipeline.start()
            pipeline.submit(np.array([5.0]), callback=lambda r: collected.append(r))
            await pipeline.drain(timeout_s=5.0)
            await asyncio.sleep(0.1)  # Let callback fire
            await pipeline.stop()

        asyncio.run(run())
        self.assertTrue(len(collected) > 0)

    def test_compute_fn_applied(self):
        async def run():
            pipeline = AsyncPipeline(compute_fn=lambda x: x ** 2)
            await pipeline.start()
            pipeline.submit(np.array([3.0]))
            await pipeline.drain(timeout_s=5.0)
            await asyncio.sleep(0.05)
            await pipeline.stop()
            return pipeline

        pipeline = asyncio.run(run())
        self.assertGreaterEqual(pipeline.total_completed, 1)

    def test_stats(self):
        async def run():
            pipeline = AsyncPipeline(compute_fn=lambda x: x)
            await pipeline.start()
            pipeline.submit(np.array([1.0]))
            await pipeline.drain(timeout_s=5.0)
            await pipeline.stop()
            return pipeline.get_stats()

        stats = asyncio.run(run())
        self.assertIn('total_submitted', stats)
        self.assertIn('total_completed', stats)
        self.assertIn('avg_latency_ms', stats)

    def test_factory(self):
        p = create_async_pipeline(compute_fn=lambda x: x, buffer_count=4)
        self.assertEqual(p.buffer_count, 4)


# ===========================================================================
# Phase 2.2: Kernel Profile Cache
# ===========================================================================

class TestClassifyShape(unittest.TestCase):
    """Test shape classification"""

    def test_small(self):
        self.assertEqual(classify_shape((10, 10)), 'small_dense')

    def test_medium(self):
        self.assertEqual(classify_shape((100, 100)), 'medium_dense')

    def test_large(self):
        self.assertEqual(classify_shape((500, 500)), 'large_dense')

    def test_empty(self):
        self.assertEqual(classify_shape(()), 'small_dense')


class TestKernelProfileCache(unittest.TestCase):
    """Test persistent kernel profile cache"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache_path = os.path.join(self.tmpdir, 'test_profiles.json')

    def tearDown(self):
        if os.path.exists(self.cache_path):
            os.unlink(self.cache_path)

    def test_update_and_get(self):
        cache = KernelProfileCache(cache_path=self.cache_path)
        cache.update('PATTERN_RECOGNITION', 'medium_dense', 'cpu_x86_64', 1.5)
        profile = cache.get('PATTERN_RECOGNITION', 'medium_dense', 'cpu_x86_64')
        self.assertIsNotNone(profile)
        self.assertAlmostEqual(profile.avg_latency_ms, 1.5)

    def test_ema_update(self):
        cache = KernelProfileCache(cache_path=self.cache_path)
        cache.update('OP', 'small_dense', 'cpu', 1.0)
        cache.update('OP', 'small_dense', 'cpu', 2.0)
        profile = cache.get('OP', 'small_dense', 'cpu')
        # EMA: 0.1*2.0 + 0.9*1.0 = 1.1
        self.assertAlmostEqual(profile.avg_latency_ms, 1.1, places=5)
        self.assertEqual(profile.execution_count, 2)

    def test_save_and_load(self):
        cache = KernelProfileCache(cache_path=self.cache_path)
        cache.update('OP', 'small_dense', 'cpu', 2.0)
        self.assertTrue(cache.save())

        cache2 = KernelProfileCache(cache_path=self.cache_path)
        loaded = cache2.load()
        self.assertEqual(loaded, 1)
        profile = cache2.get('OP', 'small_dense', 'cpu')
        self.assertIsNotNone(profile)
        self.assertAlmostEqual(profile.avg_latency_ms, 2.0)

    def test_load_missing_file(self):
        cache = KernelProfileCache(cache_path='/nonexistent/path.json')
        self.assertEqual(cache.load(), 0)

    def test_best_architecture(self):
        cache = KernelProfileCache(cache_path=self.cache_path)
        cache.update('OP', 'medium_dense', 'cpu_x86_64', 5.0)
        cache.update('OP', 'medium_dense', 'gpu_cuda', 0.5)
        best = cache.best_architecture('OP', 'medium_dense')
        self.assertEqual(best, 'gpu_cuda')

    def test_warmup(self):
        cache = create_kernel_cache(cache_path=self.cache_path)
        self.assertTrue(cache._warmed_up)
        self.assertGreater(len(cache._warmup_patterns), 0)

    def test_stats(self):
        cache = KernelProfileCache(cache_path=self.cache_path)
        cache.update('OP', 'small_dense', 'cpu', 1.0)
        stats = cache.get_stats()
        self.assertEqual(stats['total_profiles'], 1)
        self.assertFalse(stats['warmed_up'])

    def test_profile_serialization_roundtrip(self):
        original = SpecializedKernelProfile(
            operation='TEST_OP',
            shape_class='medium_dense',
            architecture='gpu_cuda',
            avg_latency_ms=1.23,
            min_latency_ms=0.5,
            max_latency_ms=3.0,
            throughput_ops_sec=813.0,
            execution_count=42,
        )
        d = original.to_dict()
        restored = SpecializedKernelProfile.from_dict(d)
        self.assertEqual(restored.operation, 'TEST_OP')
        self.assertAlmostEqual(restored.avg_latency_ms, 1.23)
        self.assertEqual(restored.execution_count, 42)


# ===========================================================================
# Phase 5.2: Bayesian Tuner
# ===========================================================================

class TestBayesianTuner(unittest.TestCase):
    """Test Bayesian-style hyperparameter optimization"""

    def setUp(self):
        self.tuner = BayesianTuner(
            bounds={'lr': (0.001, 1.0), 'batch': (1.0, 256.0)},
            rng=np.random.default_rng(42),
        )

    def test_first_suggest_returns_center(self):
        params = self.tuner.suggest()
        self.assertAlmostEqual(params['lr'], 0.5005, places=3)
        self.assertAlmostEqual(params['batch'], 128.5, places=1)

    def test_observe_updates_best(self):
        self.tuner.observe({'lr': 0.01, 'batch': 32}, 0.9)
        self.tuner.observe({'lr': 0.1, 'batch': 64}, 0.95)
        self.assertEqual(self.tuner.best_objective, 0.95)
        self.assertAlmostEqual(self.tuner.best_params['lr'], 0.1)

    def test_suggest_stays_in_bounds(self):
        # Seed with some observations
        for i in range(10):
            self.tuner.observe(
                {'lr': 0.01 * (i + 1), 'batch': 32.0 * (i + 1)},
                objective=float(i) / 10,
            )
        for _ in range(20):
            params = self.tuner.suggest()
            self.assertGreaterEqual(params['lr'], 0.001)
            self.assertLessEqual(params['lr'], 1.0)
            self.assertGreaterEqual(params['batch'], 1.0)
            self.assertLessEqual(params['batch'], 256.0)

    def test_converges_toward_optimum(self):
        # Simple quadratic objective: peak at lr=0.5, batch=128
        def objective(params):
            return -((params['lr'] - 0.5) ** 2 + (params['batch'] - 128) ** 2 / 16384)

        for _ in range(30):
            params = self.tuner.suggest()
            self.tuner.observe(params, objective(params))

        # Best should be reasonably close to the optimum
        best = self.tuner.best_params
        self.assertLess(abs(best['lr'] - 0.5), 0.3)

    def test_stats(self):
        self.tuner.observe({'lr': 0.01, 'batch': 32}, 0.8)
        stats = self.tuner.get_stats()
        self.assertEqual(stats['total_observations'], 1)
        self.assertIn('best_params', stats)

    def test_factory(self):
        t = create_bayesian_tuner({'x': (0, 1)})
        self.assertIsInstance(t, BayesianTuner)


# ===========================================================================
# Phase 5.2: Workload Clusterer
# ===========================================================================

class TestWorkloadClusterer(unittest.TestCase):
    """Test online workload characterization"""

    def setUp(self):
        self.clusterer = WorkloadClusterer(distance_threshold=1.5)

    def test_first_observation_creates_cluster(self):
        features = WorkloadClusterer.extract_features((100, 100))
        idx = self.clusterer.assign(features, latency_ms=1.0)
        self.assertEqual(idx, 0)
        self.assertEqual(len(self.clusterer.clusters), 1)

    def test_similar_workloads_same_cluster(self):
        f1 = WorkloadClusterer.extract_features((100, 100), 1.0)
        f2 = WorkloadClusterer.extract_features((110, 110), 1.1)
        idx1 = self.clusterer.assign(f1, 1.0)
        idx2 = self.clusterer.assign(f2, 1.1)
        self.assertEqual(idx1, idx2)

    def test_dissimilar_workloads_different_clusters(self):
        f1 = WorkloadClusterer.extract_features((10, 10), 0.1)
        f2 = WorkloadClusterer.extract_features((1000, 1000), 100.0)
        self.clusterer.assign(f1, 0.1)
        idx2 = self.clusterer.assign(f2, 100.0)
        self.assertEqual(len(self.clusterer.clusters), 2)

    def test_characterize(self):
        result = self.clusterer.characterize((50, 50), latency_ms=2.0)
        self.assertIn('cluster_id', result)
        self.assertIn('cluster_label', result)
        self.assertIn('features', result)
        self.assertEqual(result['total_clusters'], 1)

    def test_stats(self):
        self.clusterer.characterize((100,), 1.0)
        self.clusterer.characterize((100,), 2.0)
        stats = self.clusterer.get_stats()
        self.assertEqual(stats['total_observations'], 2)
        self.assertGreaterEqual(stats['num_clusters'], 1)

    def test_factory(self):
        c = create_workload_clusterer(distance_threshold=2.0)
        self.assertEqual(c.distance_threshold, 2.0)


# ===========================================================================
# Phase 5.2: Tradeoff Optimizer
# ===========================================================================

class TestTradeoffOptimizer(unittest.TestCase):
    """Test memory/compute tradeoff optimization"""

    def setUp(self):
        self.optimizer = TradeoffOptimizer(
            memory_budget_mb=512.0, latency_budget_ms=5.0
        )

    def test_observe_and_pareto(self):
        self.optimizer.observe(100, 5.0, {'precision': 32})
        self.optimizer.observe(200, 2.0, {'precision': 16})
        self.optimizer.observe(400, 1.0, {'precision': 8})
        front = self.optimizer.pareto_front()
        self.assertEqual(len(front), 3)  # All are Pareto-optimal

    def test_dominated_points_excluded(self):
        self.optimizer.observe(100, 5.0, {'cfg': 'a'})
        self.optimizer.observe(200, 6.0, {'cfg': 'b'})  # Dominated
        self.optimizer.observe(300, 3.0, {'cfg': 'c'})
        front = self.optimizer.pareto_front()
        self.assertEqual(len(front), 2)  # 'b' is dominated

    def test_recommend_within_budget(self):
        self.optimizer.observe(100, 4.0, {'precision': 32})
        self.optimizer.observe(600, 1.0, {'precision': 8})  # Over memory budget
        rec = self.optimizer.recommend()
        self.assertIsNotNone(rec)
        self.assertEqual(rec['precision'], 32)

    def test_recommend_empty(self):
        self.assertIsNone(self.optimizer.recommend())

    def test_stats(self):
        self.optimizer.observe(100, 2.0, {'a': 1})
        stats = self.optimizer.get_stats()
        self.assertEqual(stats['total_observations'], 1)
        self.assertEqual(stats['pareto_front_size'], 1)

    def test_factory(self):
        t = create_tradeoff_optimizer(memory_budget_mb=256, latency_budget_ms=10)
        self.assertEqual(t.memory_budget_mb, 256)
        self.assertEqual(t.latency_budget_ms, 10)


if __name__ == '__main__':
    unittest.main()
