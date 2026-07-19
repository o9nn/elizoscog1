#!/usr/bin/env python3
"""
Tests for BEASTMODE optimization components:
- Hardware feature detection
- Aligned/arena memory allocation
- Kernel fusion pipeline
- Thompson Sampling bandit selection
- Dynamic per-channel quantization
- HDR latency histogram
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from beastmode.hardware import (
    detect_cpu_features, detect_gpu_capabilities, recommend_backend, CPUFeatures
)
from beastmode.accelerators import (
    aligned_empty, is_aligned, ArenaAllocator, SIMDAccelerator,
    TensorCompressor, CompressionConfig
)
from beastmode.kernel_fusion import (
    FusionPipeline, OperationGraph, fused_multiply_add,
    fused_matmul_bias_relu, fused_softmax, create_fusion_pipeline
)
from beastmode.bandit import (
    ThompsonSamplingSelector, LatencyRewardModel, DecaySchedule,
    create_thompson_selector
)
from beastmode.metrics import LatencyHistogram


class TestHardwareDetection(unittest.TestCase):
    """Test hardware feature detection"""

    def test_detect_cpu_features(self):
        cpu = detect_cpu_features()
        self.assertIsInstance(cpu, CPUFeatures)
        self.assertGreaterEqual(cpu.cpu_count, 1)
        self.assertGreaterEqual(cpu.numa_nodes, 1)
        self.assertIn(cpu.simd_vector_width, (1, 4, 8, 16))
        self.assertIn(cpu.optimal_alignment, (16, 32, 64))

    def test_features_cached(self):
        self.assertIs(detect_cpu_features(), detect_cpu_features())

    def test_to_dict(self):
        d = detect_cpu_features().to_dict()
        for key in ('machine', 'features', 'simd_vector_width', 'optimal_alignment'):
            self.assertIn(key, d)

    def test_gpu_capabilities(self):
        gpu = detect_gpu_capabilities()
        self.assertIn('cuda_available', gpu)
        self.assertIn('opencl_available', gpu)

    def test_recommend_backend(self):
        rec = recommend_backend()
        self.assertIn(rec['primary'], rec['fallback_chain'])
        self.assertGreaterEqual(len(rec['fallback_chain']), 1)


class TestAlignedAllocation(unittest.TestCase):
    """Test aligned memory allocation"""

    def test_aligned_empty_alignment(self):
        for alignment in (16, 32, 64):
            arr = aligned_empty((100, 100), np.float32, alignment)
            self.assertTrue(is_aligned(arr, alignment))
            self.assertEqual(arr.shape, (100, 100))
            self.assertEqual(arr.dtype, np.float32)

    def test_aligned_empty_writable(self):
        arr = aligned_empty((64,), np.float32, 64)
        arr[:] = 1.0
        self.assertTrue(np.all(arr == 1.0))


class TestArenaAllocator(unittest.TestCase):
    """Test arena-style memory allocation"""

    def setUp(self):
        self.arena = ArenaAllocator(capacity_mb=1.0, alignment=64)

    def test_basic_allocation(self):
        t = self.arena.allocate((10, 10), np.float32)
        self.assertEqual(t.shape, (10, 10))
        self.assertTrue(is_aligned(t, 64))

    def test_allocations_do_not_overlap(self):
        a = self.arena.allocate((100,), np.float32)
        b = self.arena.allocate((100,), np.float32)
        a[:] = 1.0
        b[:] = 2.0
        self.assertTrue(np.all(a == 1.0))
        self.assertTrue(np.all(b == 2.0))

    def test_reset_reuses_memory(self):
        self.arena.allocate((100,), np.float32)
        used_before = self.arena.get_stats()['used_mb']
        self.arena.reset()
        self.assertEqual(self.arena.get_stats()['used_mb'], 0.0)
        self.assertGreater(used_before, 0.0)

    def test_overflow_fallback(self):
        # Request more than 1MB capacity
        big = self.arena.allocate((1024, 1024), np.float32)  # 4MB
        self.assertEqual(big.shape, (1024, 1024))
        self.assertEqual(self.arena.get_stats()['overflow_count'], 1)

    def test_stats(self):
        self.arena.allocate((10,), np.float32)
        stats = self.arena.get_stats()
        self.assertEqual(stats['allocation_count'], 1)
        self.assertGreater(stats['used_mb'], 0.0)


class TestSIMDAcceleratorEnhanced(unittest.TestCase):
    """Test enhanced SIMD accelerator with real feature detection"""

    def setUp(self):
        self.simd = SIMDAccelerator()

    def test_real_feature_detection(self):
        # cpu_info should come from real detection, not the timing heuristic
        self.assertIn('has_fma', self.simd.cpu_info)
        self.assertIn('numa_nodes', self.simd.cpu_info)

    def test_vectorized_add_aligned_output(self):
        a = np.random.randn(1000).astype(np.float32)
        b = np.random.randn(1000).astype(np.float32)
        result = self.simd.vectorized_add(a, b)
        np.testing.assert_allclose(result, a + b, rtol=1e-6)
        self.assertTrue(is_aligned(result, self.simd.config.alignment))

    def test_vectorized_fma(self):
        a = np.random.randn(500).astype(np.float32)
        b = np.random.randn(500).astype(np.float32)
        c = np.random.randn(500).astype(np.float32)
        result = self.simd.vectorized_fma(a, b, c)
        np.testing.assert_allclose(result, a * b + c, rtol=1e-5)


class TestKernelFusion(unittest.TestCase):
    """Test kernel fusion pipeline"""

    def test_fused_multiply_add(self):
        a, b, c = (np.random.randn(100).astype(np.float32) for _ in range(3))
        np.testing.assert_allclose(fused_multiply_add(a, b, c), a * b + c, rtol=1e-5)

    def test_fused_matmul_bias_relu(self):
        x = np.random.randn(4, 8).astype(np.float32)
        w = np.random.randn(8, 16).astype(np.float32)
        bias = np.random.randn(16).astype(np.float32)
        expected = np.maximum(x @ w + bias, 0)
        np.testing.assert_allclose(fused_matmul_bias_relu(x, w, bias), expected, rtol=1e-5)

    def test_fused_softmax(self):
        x = np.random.randn(10, 20).astype(np.float32)
        result = fused_softmax(x)
        np.testing.assert_allclose(result.sum(axis=-1), np.ones(10), rtol=1e-5)
        self.assertTrue(np.all(result >= 0))

    def test_fused_softmax_stability(self):
        # Large values should not overflow
        x = np.array([1000.0, 1000.0], dtype=np.float32)
        result = fused_softmax(x)
        self.assertFalse(np.any(np.isnan(result)))
        np.testing.assert_allclose(result, [0.5, 0.5], rtol=1e-5)

    def test_operation_graph_finds_fusion(self):
        graph = OperationGraph()
        graph.add_operation('matmul', other=None)
        graph.add_operation('add', other=None)
        graph.add_operation('relu')
        groups = graph.find_fusion_groups()
        self.assertEqual(len(groups), 1)
        self.assertEqual((groups[0][0], groups[0][1]), (0, 3))

    def test_pipeline_execute_fused_sequence(self):
        pipeline = create_fusion_pipeline()
        x = np.random.randn(4, 8).astype(np.float32)
        w = np.random.randn(8, 16).astype(np.float32)
        bias = np.random.randn(16).astype(np.float32)
        result = pipeline.execute(x, [
            ('matmul', {'other': w}),
            ('add', {'other': bias}),
            ('relu', {}),
        ])
        expected = np.maximum(x @ w + bias, 0)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        stats = pipeline.get_stats()
        self.assertEqual(stats['operations_fused'], 3)
        self.assertEqual(stats['fusion_groups'], 1)

    def test_pipeline_execute_unfused_ops(self):
        pipeline = FusionPipeline()
        x = np.random.randn(10).astype(np.float32)
        result = pipeline.execute(x, [('relu', {})])
        np.testing.assert_allclose(result, np.maximum(x, 0))
        self.assertEqual(pipeline.get_stats()['operations_fused'], 0)

    def test_pipeline_mixed_sequence(self):
        pipeline = FusionPipeline()
        x = np.random.randn(8).astype(np.float32)
        other = np.random.randn(8).astype(np.float32)
        result = pipeline.execute(x, [
            ('relu', {}),
            ('multiply', {'other': other}),
            ('add', {'other': other}),
        ])
        expected = np.maximum(x, 0) * other + other
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestThompsonSampling(unittest.TestCase):
    """Test Thompson Sampling bandit selection"""

    def setUp(self):
        self.selector = ThompsonSamplingSelector(
            rng=np.random.default_rng(42))

    def test_forced_exploration_of_new_arms(self):
        arms = ['cpu', 'gpu']
        selected = {self.selector.select('op1', arms) for _ in range(4)}
        for _ in range(10):
            arm = self.selector.select('op1', arms)
            self.selector.update('op1', arm, 0.5)
        # Both arms should have been tried
        self.assertEqual(len({k[1] for k in self.selector.arms}), 2)

    def test_converges_to_best_arm(self):
        arms = ['slow', 'fast']
        rng = np.random.default_rng(0)
        for _ in range(300):
            arm = self.selector.select('op', arms)
            reward = 0.9 if arm == 'fast' else 0.2
            reward += rng.normal(0, 0.05)
            self.selector.update('op', arm, reward)
        self.assertEqual(self.selector.best_arm('op', arms), 'fast')

    def test_contextual_independence(self):
        arms = ['a', 'b']
        for _ in range(100):
            arm = self.selector.select('ctx1', arms)
            self.selector.update('ctx1', arm, 0.9 if arm == 'a' else 0.1)
            arm = self.selector.select('ctx2', arms)
            self.selector.update('ctx2', arm, 0.9 if arm == 'b' else 0.1)
        self.assertEqual(self.selector.best_arm('ctx1', arms), 'a')
        self.assertEqual(self.selector.best_arm('ctx2', arms), 'b')

    def test_exploration_decay(self):
        schedule = DecaySchedule(initial_rate=0.2, min_rate=0.01, half_life=100)
        self.assertAlmostEqual(schedule.rate_at(0), 0.2)
        self.assertAlmostEqual(schedule.rate_at(100), 0.1)
        self.assertEqual(schedule.rate_at(10**6), 0.01)

    def test_empty_arms_raises(self):
        with self.assertRaises(ValueError):
            self.selector.select('op', [])

    def test_factory(self):
        selector = create_thompson_selector(initial_exploration=0.3)
        self.assertEqual(selector.decay.initial_rate, 0.3)

    def test_stats(self):
        arms = ['x']
        arm = self.selector.select('op', arms)
        self.selector.update('op', arm, 0.5)
        stats = self.selector.get_stats()
        self.assertEqual(stats['total_selections'], 1)
        self.assertEqual(stats['contexts_tracked'], 1)


class TestLatencyRewardModel(unittest.TestCase):
    """Test latency reward model"""

    def test_reward_range(self):
        model = LatencyRewardModel(target_latency_ms=5.0)
        fast = model.compute_reward(latency_ms=1.0, accuracy=1.0)
        slow = model.compute_reward(latency_ms=50.0, accuracy=1.0)
        self.assertGreater(fast, slow)
        self.assertLessEqual(fast, 1.0)
        self.assertGreaterEqual(slow, 0.0)

    def test_regret_tracking(self):
        model = LatencyRewardModel()
        model.compute_reward(1.0, 1.0)
        model.compute_reward(100.0, 1.0)
        self.assertGreater(model.cumulative_regret, 0.0)


class TestDynamicQuantization(unittest.TestCase):
    """Test dynamic per-channel quantization"""

    def setUp(self):
        self.compressor = TensorCompressor(CompressionConfig(per_channel=True))

    def test_per_channel_roundtrip(self):
        # Channels with very different scales
        data = np.stack([
            np.random.randn(64).astype(np.float32) * 0.01,
            np.random.randn(64).astype(np.float32) * 100.0,
        ])
        q, scales, mins = self.compressor.quantize_per_channel(data)
        restored = self.compressor.dequantize_per_channel(q, scales, mins)
        # Per-channel error should be small relative to each channel's range
        for ch in range(2):
            ch_range = data[ch].max() - data[ch].min()
            max_err = np.abs(data[ch] - restored[ch]).max()
            self.assertLess(max_err, ch_range / 100)

    def test_per_channel_beats_global(self):
        data = np.stack([
            np.ones(64, dtype=np.float32) * 0.001,
            np.random.randn(64).astype(np.float32) * 1000.0,
        ])
        q, scales, mins = self.compressor.quantize_per_channel(data)
        restored_pc = self.compressor.dequantize_per_channel(q, scales, mins)
        packed = self.compressor._quantize_8bit(data)
        restored_global = self.compressor._dequantize_8bit(packed, data.shape, np.float32)
        err_pc = np.abs(data[0] - restored_pc[0]).max()
        err_global = np.abs(data[0] - restored_global[0]).max()
        self.assertLessEqual(err_pc, err_global)

    def test_select_precision_easy_tensor(self):
        data = np.random.uniform(0, 1, (4, 64)).astype(np.float32)
        bits = self.compressor.select_precision(data, target_accuracy=0.95)
        self.assertIn(bits, (8, 16, 32))

    def test_select_precision_strict_accuracy(self):
        data = np.random.randn(4, 64).astype(np.float32)
        loose = self.compressor.select_precision(data, target_accuracy=0.5)
        strict = self.compressor.select_precision(data, target_accuracy=0.9999999)
        self.assertLessEqual(loose, strict)

    def test_select_precision_zero_tensor(self):
        data = np.zeros((2, 8), dtype=np.float32)
        self.assertEqual(self.compressor.select_precision(data), 8)


class TestLatencyHistogram(unittest.TestCase):
    """Test HDR-style latency histogram"""

    def setUp(self):
        self.hist = LatencyHistogram()

    def test_record_and_count(self):
        for v in (0.5, 1.0, 2.0, 4.0):
            self.hist.record(v)
        self.assertEqual(self.hist.total_count, 4)
        self.assertAlmostEqual(self.hist.mean, 1.875)

    def test_percentile_accuracy(self):
        rng = np.random.default_rng(7)
        samples = rng.uniform(1.0, 10.0, 10000)
        for s in samples:
            self.hist.record(s)
        p50 = self.hist.percentile(50)
        true_p50 = np.percentile(samples, 50)
        # Log-bucket estimate should be within 15% of the true value
        self.assertLess(abs(p50 - true_p50) / true_p50, 0.15)

    def test_percentile_ordering(self):
        for v in np.random.uniform(0.1, 100.0, 1000):
            self.hist.record(v)
        self.assertLessEqual(self.hist.percentile(50), self.hist.percentile(95))
        self.assertLessEqual(self.hist.percentile(95), self.hist.percentile(99))

    def test_empty_histogram(self):
        self.assertEqual(self.hist.percentile(99), 0.0)
        self.assertEqual(self.hist.get_summary()['count'], 0)

    def test_out_of_range_clamped(self):
        self.hist.record(1e-9)
        self.hist.record(1e9)
        self.assertEqual(self.hist.total_count, 2)

    def test_reset(self):
        self.hist.record(5.0)
        self.hist.reset()
        self.assertEqual(self.hist.total_count, 0)

    def test_summary_keys(self):
        self.hist.record(1.0)
        summary = self.hist.get_summary()
        for key in ('count', 'mean_ms', 'p50_ms', 'p95_ms', 'p99_ms', 'p999_ms'):
            self.assertIn(key, summary)

    def test_constant_memory(self):
        buckets_before = self.hist.counts.nbytes
        for v in np.random.uniform(0.1, 1000.0, 50000):
            self.hist.record(v)
        self.assertEqual(self.hist.counts.nbytes, buckets_before)


if __name__ == '__main__':
    unittest.main()
