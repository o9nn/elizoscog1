#!/usr/bin/env python3
"""
Tests for BeastMode Inference Engine
====================================

Comprehensive test suite for the high-performance inference engine,
including self-optimizing kernel selection, batch processing, and
parallel execution.
"""

import asyncio
import unittest
import numpy as np
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from beastmode.inference_engine import (
    BeastModeInferenceEngine,
    KernelSelector,
    AdaptiveOptimizer,
    BatchProcessor,
    ParallelExecutor,
    OptimizationStrategy,
    InputCharacteristics,
    get_beastmode_engine,
    beast_infer,
    beast_batch_infer,
    beast_parallel_infer
)

from beastmode.accelerators import (
    SIMDAccelerator,
    MemoryOptimizer,
    CacheManager,
    TensorCompressor
)

from beastmode.metrics import (
    PerformanceTracker,
    LatencyProfiler,
    ThroughputMonitor,
    ResourceAnalyzer,
    MetricType
)

from src.core.ggml_symbolic_kernels import (
    SymbolicTensor, SymbolicOperation, KernelArchitecture
)


class TestKernelSelector(unittest.TestCase):
    """Test self-optimizing kernel selection"""
    
    def setUp(self):
        self.selector = KernelSelector(OptimizationStrategy.ADAPTIVE)
    
    def test_input_classification(self):
        """Test input tensor classification"""
        # Small tensor
        small_tensor = SymbolicTensor(
            data=np.array([1.0, 2.0, 3.0]),
            symbols={'a': 1}
        )
        result = self.selector.classify_input([small_tensor])
        self.assertEqual(result, InputCharacteristics.SMALL_DENSE)
        
        # Medium tensor
        medium_tensor = SymbolicTensor(
            data=np.random.randn(5000).astype(np.float32),
            symbols={'a': 1}
        )
        result = self.selector.classify_input([medium_tensor])
        self.assertEqual(result, InputCharacteristics.MEDIUM_DENSE)
    
    def test_kernel_selection(self):
        """Test kernel selection algorithm"""
        tensor = SymbolicTensor(
            data=np.array([1.0, 2.0, 3.0]),
            symbols={'a': 1}
        )
        
        available_archs = [KernelArchitecture.CPU_X86_64]
        selected = self.selector.select_kernel(
            SymbolicOperation.SYMBOL_ADD,
            [tensor],
            available_archs
        )
        
        self.assertIn(selected, available_archs)
    
    def test_profile_update(self):
        """Test kernel profile updates"""
        self.selector.update_profile(
            SymbolicOperation.SYMBOL_ADD,
            KernelArchitecture.CPU_X86_64,
            InputCharacteristics.SMALL_DENSE,
            0.5,
            True
        )
        
        profile_key = "SYMBOL_ADD_cpu_x86_64_small_dense"
        self.assertIn(profile_key, self.selector.profiles)
        
        profile = self.selector.profiles[profile_key]
        self.assertEqual(profile.execution_count, 1)
        self.assertGreater(profile.avg_latency_ms, 0)


class TestAdaptiveOptimizer(unittest.TestCase):
    """Test adaptive optimization system"""
    
    def setUp(self):
        self.optimizer = AdaptiveOptimizer(target_latency_ms=5.0)
    
    def test_optimization_recommendations(self):
        """Test optimization recommendation generation"""
        tensor = SymbolicTensor(
            data=np.array([1.0, 2.0, 3.0]),
            symbols={'a': 1}
        )
        
        # Fast operation - should maintain
        recommendations = self.optimizer.optimize_execution(
            SymbolicOperation.SYMBOL_ADD,
            [tensor],
            current_latency_ms=1.0
        )
        self.assertEqual(recommendations['action'], 'maintain')
        
        # Slow operation - should optimize
        recommendations = self.optimizer.optimize_execution(
            SymbolicOperation.SYMBOL_ADD,
            [tensor],
            current_latency_ms=30.0
        )
        self.assertEqual(recommendations['action'], 'aggressive_optimize')
    
    def test_batch_size_tuning(self):
        """Test automatic batch size tuning"""
        new_batch = self.optimizer.auto_tune_batch_size(
            SymbolicOperation.SYMBOL_ADD,
            current_batch_size=32,
            latency_ms=10.0,
            throughput=3200.0
        )
        
        self.assertIsInstance(new_batch, int)
        self.assertGreaterEqual(new_batch, 1)


class TestBeastModeInferenceEngine(unittest.TestCase):
    """Test the main BeastMode inference engine"""
    
    def setUp(self):
        # Create fresh engine instance
        import beastmode.inference_engine as engine_module
        engine_module._beastmode_engine = None
        self.engine = get_beastmode_engine()
    
    def test_basic_inference(self):
        """Test basic symbolic inference"""
        async def run_test():
            tensor1 = SymbolicTensor(
                data=np.array([1.0, 2.0, 3.0]),
                symbols={'a': 1}
            )
            tensor2 = SymbolicTensor(
                data=np.array([4.0, 5.0, 6.0]),
                symbols={'b': 2}
            )
            
            result = await self.engine.infer(
                SymbolicOperation.SYMBOL_ADD,
                [tensor1, tensor2]
            )
            
            expected = np.array([5.0, 7.0, 9.0])
            np.testing.assert_array_almost_equal(result.data, expected)
            
            return result
        
        result = asyncio.run(run_test())
        self.assertEqual(self.engine.total_operations, 1)
    
    def test_batch_inference(self):
        """Test batch inference"""
        async def run_test():
            operations = []
            for i in range(5):
                t1 = SymbolicTensor(data=np.array([float(i)]), symbols={'i': i})
                t2 = SymbolicTensor(data=np.array([float(i)]), symbols={'j': i})
                operations.append((SymbolicOperation.SYMBOL_ADD, [t1, t2]))
            
            results = await self.engine.batch_infer(operations)
            
            self.assertEqual(len(results), 5)
            
            return results
        
        results = asyncio.run(run_test())
        for i, result in enumerate(results):
            expected = float(i) + float(i)
            self.assertAlmostEqual(result.data[0], expected, places=5)
    
    def test_parallel_inference(self):
        """Test parallel inference"""
        async def run_test():
            named_ops = {
                'op1': (SymbolicOperation.SYMBOL_ADD, [
                    SymbolicTensor(data=np.array([1.0]), symbols={'a': 1}),
                    SymbolicTensor(data=np.array([2.0]), symbols={'b': 2})
                ]),
                'op2': (SymbolicOperation.SYMBOL_MULTIPLY, [
                    SymbolicTensor(data=np.array([3.0]), symbols={'c': 3}),
                    SymbolicTensor(data=np.array([4.0]), symbols={'d': 4})
                ])
            }
            
            results = await self.engine.parallel_infer(named_ops)
            
            self.assertIn('op1', results)
            self.assertIn('op2', results)
            self.assertAlmostEqual(results['op1'].data[0], 3.0, places=5)
            self.assertAlmostEqual(results['op2'].data[0], 12.0, places=5)
            
            return results
        
        results = asyncio.run(run_test())
        self.assertEqual(len(results), 2)
    
    def test_caching(self):
        """Test result caching"""
        async def run_test():
            tensor1 = SymbolicTensor(data=np.array([1.0]), symbols={'a': 1})
            tensor2 = SymbolicTensor(data=np.array([2.0]), symbols={'b': 2})
            
            # First call - cache miss
            result1 = await self.engine.infer(
                SymbolicOperation.SYMBOL_ADD,
                [tensor1, tensor2]
            )
            
            # Second call with same inputs - cache hit
            result2 = await self.engine.infer(
                SymbolicOperation.SYMBOL_ADD,
                [tensor1, tensor2]
            )
            
            self.assertGreater(self.engine.cache_hits, 0)
            np.testing.assert_array_equal(result1.data, result2.data)
            
            return result1, result2
        
        asyncio.run(run_test())
    
    def test_performance_report(self):
        """Test performance reporting"""
        async def run_test():
            tensor1 = SymbolicTensor(data=np.array([1.0]), symbols={'a': 1})
            tensor2 = SymbolicTensor(data=np.array([2.0]), symbols={'b': 2})
            
            for _ in range(10):
                await self.engine.infer(
                    SymbolicOperation.SYMBOL_ADD,
                    [tensor1, tensor2]
                )
            
            report = self.engine.get_performance_report()
            
            self.assertIn('total_operations', report)
            self.assertIn('avg_latency_ms', report)
            self.assertIn('cache_hit_rate', report)
            self.assertIn('recent_performance', report)
            
            return report
        
        report = asyncio.run(run_test())
        self.assertGreater(report['total_operations'], 0)


class TestSIMDAccelerator(unittest.TestCase):
    """Test SIMD acceleration"""
    
    def setUp(self):
        self.accelerator = SIMDAccelerator()
    
    def test_vectorized_add(self):
        """Test vectorized addition"""
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        
        result = self.accelerator.vectorized_add(a, b)
        expected = np.array([6.0, 8.0, 10.0, 12.0], dtype=np.float32)
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_vectorized_fma(self):
        """Test fused multiply-add"""
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        c = np.array([5.0, 6.0], dtype=np.float32)
        
        result = self.accelerator.vectorized_fma(a, b, c)
        expected = a * b + c
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_batch_process(self):
        """Test batch tensor processing"""
        tensors = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0])
        ]
        
        results = self.accelerator.batch_process(tensors, 'relu')
        
        self.assertEqual(len(results), 2)
        np.testing.assert_array_almost_equal(results[0], tensors[0])


class TestCacheManager(unittest.TestCase):
    """Test caching system"""
    
    def setUp(self):
        self.cache = CacheManager()
    
    def test_cache_put_get(self):
        """Test basic cache operations"""
        tensor = SymbolicTensor(data=np.array([1.0, 2.0]), symbols={'a': 1})
        
        self.cache.put('key1', tensor)
        result = self.cache.get('key1')
        
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result.data, tensor.data)
    
    def test_cache_miss(self):
        """Test cache miss"""
        result = self.cache.get('nonexistent')
        self.assertIsNone(result)
    
    def test_cache_eviction(self):
        """Test LRU eviction"""
        self.cache.config.max_size = 3
        
        for i in range(5):
            tensor = SymbolicTensor(data=np.array([float(i)]), symbols={'i': i})
            self.cache.put(f'key{i}', tensor)
        
        # First entries should be evicted
        self.assertIsNone(self.cache.get('key0'))
        self.assertIsNone(self.cache.get('key1'))
        
        # Later entries should still exist
        self.assertIsNotNone(self.cache.get('key4'))
    
    def test_cache_stats(self):
        """Test cache statistics"""
        tensor = SymbolicTensor(data=np.array([1.0]), symbols={'a': 1})
        
        self.cache.put('key1', tensor)
        self.cache.get('key1')
        self.cache.get('key1')
        self.cache.get('nonexistent')
        
        stats = self.cache.get_stats()
        
        self.assertEqual(stats['hits'], 2)
        self.assertEqual(stats['misses'], 1)
        self.assertGreater(stats['hit_rate'], 0.5)


class TestTensorCompressor(unittest.TestCase):
    """Test tensor compression"""
    
    def setUp(self):
        self.compressor = TensorCompressor()
    
    def test_compression_decompression(self):
        """Test round-trip compression"""
        tensor = SymbolicTensor(
            data=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            symbols={'a': 1}
        )
        
        compressed, metadata = self.compressor.compress(tensor)
        decompressed = self.compressor.decompress(compressed, metadata)
        
        # Should approximately equal (quantization may lose precision)
        np.testing.assert_array_almost_equal(
            tensor.data, 
            decompressed.data, 
            decimal=1  # 8-bit quantization has limited precision
        )
    
    def test_compression_ratio(self):
        """Test compression effectiveness"""
        tensor = SymbolicTensor(
            data=np.random.randn(1000).astype(np.float32),
            symbols={'a': 1}
        )
        
        compressed, metadata = self.compressor.compress(tensor)
        
        self.assertGreater(metadata['compression_ratio'], 1.0)


class TestPerformanceTracker(unittest.TestCase):
    """Test performance tracking"""
    
    def setUp(self):
        self.tracker = PerformanceTracker()
    
    def test_metric_recording(self):
        """Test metric recording and retrieval"""
        for i in range(10):
            self.tracker.record(MetricType.LATENCY, float(i))
        
        stats = self.tracker.get_stats(MetricType.LATENCY)
        
        self.assertEqual(stats['count'], 10)
        self.assertAlmostEqual(stats['mean'], 4.5, places=1)
    
    def test_baseline_comparison(self):
        """Test baseline comparison"""
        self.tracker.set_baseline(MetricType.LATENCY, 10.0)
        
        for _ in range(10):
            self.tracker.record(MetricType.LATENCY, 5.0)
        
        improvement = self.tracker.get_improvement(MetricType.LATENCY)
        
        self.assertIsNotNone(improvement)
        self.assertAlmostEqual(improvement, 0.5, places=1)  # 50% improvement


class TestLatencyProfiler(unittest.TestCase):
    """Test latency profiling"""
    
    def setUp(self):
        self.profiler = LatencyProfiler(sla_ms=10.0)
    
    def test_operation_timing(self):
        """Test operation timing"""
        start = self.profiler.start_operation('op1')
        time.sleep(0.001)  # Simulate work
        latency = self.profiler.end_operation('op1', 'test_op', start)
        
        self.assertGreater(latency, 0)
        
        stats = self.profiler.get_operation_stats('test_op')
        self.assertEqual(stats['count'], 1)
    
    def test_sla_compliance(self):
        """Test SLA tracking"""
        # Record operations under and over SLA
        for _ in range(5):
            start = self.profiler.start_operation('op')
            self.profiler.end_operation('op', 'test_op', start)  # Fast - under SLA
        
        compliance = self.profiler.get_sla_compliance()
        
        self.assertEqual(compliance['total_operations'], 5)
        self.assertEqual(compliance['compliance_rate'], 1.0)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""
    
    def test_beast_infer(self):
        """Test beast_infer convenience function"""
        async def run_test():
            tensor1 = SymbolicTensor(data=np.array([1.0]), symbols={'a': 1})
            tensor2 = SymbolicTensor(data=np.array([2.0]), symbols={'b': 2})
            
            result = await beast_infer(
                SymbolicOperation.SYMBOL_ADD,
                [tensor1, tensor2]
            )
            
            self.assertAlmostEqual(result.data[0], 3.0)
            
            return result
        
        asyncio.run(run_test())


class TestPerformanceTargets(unittest.TestCase):
    """Test that performance targets are met"""
    
    def test_sub_5ms_latency(self):
        """Test sub-5ms inference latency for standard operations"""
        async def run_test():
            engine = BeastModeInferenceEngine()
            
            tensor1 = SymbolicTensor(
                data=np.random.randn(1000).astype(np.float32),
                symbols={'a': 1}
            )
            tensor2 = SymbolicTensor(
                data=np.random.randn(1000).astype(np.float32),
                symbols={'b': 2}
            )
            
            # Warm up
            await engine.infer(SymbolicOperation.SYMBOL_ADD, [tensor1, tensor2])
            
            # Measure latency
            latencies = []
            for _ in range(100):
                start = time.perf_counter()
                await engine.infer(SymbolicOperation.SYMBOL_ADD, [tensor1, tensor2])
                latency_ms = (time.perf_counter() - start) * 1000
                latencies.append(latency_ms)
            
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            
            print(f"\nPerformance: avg={avg_latency:.3f}ms, p95={p95_latency:.3f}ms")
            
            # Target: sub-5ms for standard operations
            self.assertLess(avg_latency, 5.0, 
                          f"Average latency {avg_latency:.3f}ms exceeds 5ms target")
            
            return avg_latency, p95_latency
        
        asyncio.run(run_test())
    
    def test_accuracy(self):
        """Test 99%+ operation accuracy"""
        async def run_test():
            engine = BeastModeInferenceEngine()
            
            correct = 0
            total = 100
            
            for _ in range(total):
                # Generate random test case
                a = np.random.randn(10).astype(np.float32)
                b = np.random.randn(10).astype(np.float32)
                
                tensor1 = SymbolicTensor(data=a, symbols={'a': 1})
                tensor2 = SymbolicTensor(data=b, symbols={'b': 2})
                
                result = await engine.infer(
                    SymbolicOperation.SYMBOL_ADD,
                    [tensor1, tensor2]
                )
                
                expected = a + b
                
                if np.allclose(result.data, expected, rtol=1e-5):
                    correct += 1
            
            accuracy = correct / total
            print(f"\nAccuracy: {accuracy*100:.1f}%")
            
            self.assertGreaterEqual(accuracy, 0.99,
                                  f"Accuracy {accuracy*100:.1f}% below 99% target")
            
            return accuracy
        
        asyncio.run(run_test())


if __name__ == '__main__':
    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)
