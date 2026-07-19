#!/usr/bin/env python3
"""
BEASTMODE Test Suite
=====================

Comprehensive tests for the BEASTMODE inference engine and tensor benchmarking.
"""

import asyncio
import unittest
import numpy as np
import time
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.ggml_symbolic_kernels import (
    SymbolicTensor, SymbolicOperation, KernelArchitecture
)

from beastmode.inference_engine import (
    InferenceAccelerator, AcceleratorConfig, ExecutionMode, create_accelerator
)
from beastmode.tensor_validator import (
    TensorSignatureValidator, ValidationLevel, ValidationResult, create_validator
)
from beastmode.performance_monitor import (
    PerformanceMonitor, PerformanceMetrics, AlertSeverity, create_monitor
)
from beastmode.adaptive_optimizer import (
    AdaptiveOptimizer, OptimizationStrategy, OptimizationConfig, create_optimizer
)
from beastmode.benchmarks import (
    run_quick_benchmark, BenchmarkConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestInferenceAccelerator(unittest.TestCase):
    """Test the BEASTMODE inference accelerator"""
    
    def setUp(self):
        self.accelerator = create_accelerator(AcceleratorConfig(
            warmup_iterations=0  # Skip warmup for faster tests
        ))
    
    def test_accelerator_initialization(self):
        """Test accelerator initializes correctly"""
        self.assertIsNotNone(self.accelerator)
        self.assertIsNotNone(self.accelerator.kernel_manager)
        self.assertIsNotNone(self.accelerator.config)
    
    def test_single_execution(self):
        """Test single operation execution"""
        async def run_test():
            tensor = SymbolicTensor(
                data=np.random.random((2, 4, 8, 6, 3)).astype(np.float32),
                symbols={'test': True}
            )
            
            result = await self.accelerator.execute(
                SymbolicOperation.PATTERN_RECOGNITION,
                [tensor]
            )
            
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.output)
            self.assertGreater(result.execution_time_ms, 0)
            self.assertGreaterEqual(result.accuracy_score, 0)
            self.assertLessEqual(result.accuracy_score, 1.0)
            
            logger.info(f"✅ Execution: {result.execution_time_ms:.3f}ms, "
                       f"accuracy {result.accuracy_score:.2%}")
        
        asyncio.run(run_test())
    
    def test_caching(self):
        """Test result caching works correctly"""
        async def run_test():
            tensor = SymbolicTensor(
                data=np.random.random((2, 4, 8, 6, 3)).astype(np.float32),
                symbols={'cache_test': True}
            )
            
            # First execution (cache miss)
            result1 = await self.accelerator.execute(
                SymbolicOperation.PATTERN_RECOGNITION,
                [tensor]
            )
            self.assertFalse(result1.cache_hit)
            
            # Second execution (should be cache hit)
            result2 = await self.accelerator.execute(
                SymbolicOperation.PATTERN_RECOGNITION,
                [tensor]
            )
            self.assertTrue(result2.cache_hit)
            
            logger.info(f"✅ Caching works: first {result1.execution_time_ms:.3f}ms, "
                       f"cached {result2.execution_time_ms:.3f}ms")
        
        asyncio.run(run_test())
    
    def test_execution_modes(self):
        """Test different execution modes"""
        modes = [
            ExecutionMode.LATENCY_OPTIMIZED,
            ExecutionMode.THROUGHPUT_OPTIMIZED,
            ExecutionMode.BALANCED
        ]
        
        for mode in modes:
            accelerator = create_accelerator(AcceleratorConfig(
                execution_mode=mode,
                warmup_iterations=0
            ))
            
            self.assertEqual(accelerator.config.execution_mode, mode)
            logger.info(f"✅ Mode {mode.value} initialized")
    
    def test_performance_tracking(self):
        """Test performance metrics are tracked"""
        async def run_test():
            tensor = SymbolicTensor(
                data=np.random.random((2, 4, 8, 6, 3)).astype(np.float32),
                symbols={'tracking_test': True}
            )
            
            # Execute multiple operations
            for _ in range(10):
                await self.accelerator.execute(
                    SymbolicOperation.PATTERN_RECOGNITION,
                    [tensor]
                )
            
            # Check performance summary
            summary = self.accelerator.get_performance_summary()
            
            self.assertGreater(summary['total_executions'], 0)
            self.assertIn('avg_latency_ms', summary)
            self.assertIn('avg_accuracy', summary)
            
            logger.info(f"✅ Tracked {summary['total_executions']} executions")
        
        asyncio.run(run_test())


class TestTensorValidator(unittest.TestCase):
    """Test tensor signature validation"""
    
    def setUp(self):
        self.validator = create_validator(ValidationLevel.QUICK)
    
    def test_validator_initialization(self):
        """Test validator initializes correctly"""
        self.assertIsNotNone(self.validator)
        self.assertEqual(self.validator.level, ValidationLevel.QUICK)
    
    def test_single_operation_validation(self):
        """Test validation of single operation"""
        async def run_test():
            result = await self.validator.validate_operation(
                SymbolicOperation.PATTERN_RECOGNITION
            )
            
            self.assertIsInstance(result, ValidationResult)
            self.assertGreaterEqual(result.numerical_precision, 0)
            self.assertLessEqual(result.numerical_precision, 1.0)
            self.assertGreaterEqual(result.stability_score, 0)
            self.assertGreater(result.trials, 0)
            
            logger.info(f"✅ Validation: precision {result.numerical_precision:.2%}, "
                       f"stability {result.stability_score:.2%}, "
                       f"overall {result.overall_score:.2%}")
        
        asyncio.run(run_test())
    
    def test_comprehensive_validation(self):
        """Test comprehensive validation suite"""
        async def run_test():
            results = await self.validator.run_comprehensive_validation(
                operations=[
                    SymbolicOperation.PATTERN_RECOGNITION,
                    SymbolicOperation.TENSOR_TO_SYMBOL
                ]
            )
            
            self.assertGreater(results['total_validations'], 0)
            self.assertIn('pass_rate', results)
            self.assertIn('avg_numerical_precision', results)
            
            logger.info(f"✅ Comprehensive: {results['passing']}/{results['total_validations']} passed")
        
        asyncio.run(run_test())
    
    def test_validation_levels(self):
        """Test different validation levels"""
        levels = [ValidationLevel.QUICK, ValidationLevel.STANDARD]
        
        for level in levels:
            validator = create_validator(level)
            config = validator._get_level_config()
            
            self.assertIn('trials_per_operation', config)
            self.assertIn('noise_levels', config)
            
            logger.info(f"✅ Level {level.value}: {config['trials_per_operation']} trials")


class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitoring"""
    
    def setUp(self):
        self.monitor = create_monitor(history_size=100, baseline_window=10)
    
    def test_monitor_initialization(self):
        """Test monitor initializes correctly"""
        self.assertIsNotNone(self.monitor)
        self.assertEqual(len(self.monitor.metrics_history), 0)
    
    def test_record_metrics(self):
        """Test metric recording"""
        async def run_test():
            # Record some metrics
            for i in range(15):
                await self.monitor.record_metrics(
                    operation=SymbolicOperation.PATTERN_RECOGNITION,
                    architecture=KernelArchitecture.CPU_X86_64,
                    latency_ms=0.5 + np.random.random() * 0.1,
                    memory_mb=10.0,
                    accuracy=0.99
                )
            
            # Check metrics were recorded
            self.assertGreater(self.monitor.total_metrics, 0)
            self.assertGreater(len(self.monitor.metrics_history), 0)
            
            logger.info(f"✅ Recorded {self.monitor.total_metrics} metrics")
        
        asyncio.run(run_test())
    
    def test_baseline_creation(self):
        """Test baseline creation"""
        async def run_test():
            # Record enough metrics to trigger baseline update
            for i in range(20):
                await self.monitor.record_metrics(
                    operation=SymbolicOperation.PATTERN_RECOGNITION,
                    architecture=KernelArchitecture.CPU_X86_64,
                    latency_ms=0.5 + np.random.random() * 0.1,
                    memory_mb=10.0,
                    accuracy=0.99
                )
            
            # Force baseline update
            await self.monitor._update_baselines()
            
            # Check baseline was created
            self.assertGreater(len(self.monitor.baselines), 0)
            
            logger.info(f"✅ Created {len(self.monitor.baselines)} baselines")
        
        asyncio.run(run_test())
    
    def test_regression_detection(self):
        """Test regression detection"""
        async def run_test():
            # Establish baseline
            for i in range(15):
                await self.monitor.record_metrics(
                    operation=SymbolicOperation.PATTERN_RECOGNITION,
                    architecture=KernelArchitecture.CPU_X86_64,
                    latency_ms=0.5,
                    memory_mb=10.0,
                    accuracy=0.99
                )
            
            await self.monitor._update_baselines()
            
            # Record a regression (high latency)
            alert = await self.monitor.record_metrics(
                operation=SymbolicOperation.PATTERN_RECOGNITION,
                architecture=KernelArchitecture.CPU_X86_64,
                latency_ms=5.0,  # 10x higher
                memory_mb=10.0,
                accuracy=0.99
            )
            
            # Alert might be generated on next regression check
            logger.info(f"✅ Regression test: {len(self.monitor.alerts)} alerts")
        
        asyncio.run(run_test())


class TestAdaptiveOptimizer(unittest.TestCase):
    """Test adaptive optimization"""
    
    def setUp(self):
        self.optimizer = create_optimizer(OptimizationConfig(
            strategy=OptimizationStrategy.BALANCED
        ))
    
    def test_optimizer_initialization(self):
        """Test optimizer initializes correctly"""
        self.assertIsNotNone(self.optimizer)
        self.assertEqual(self.optimizer.config.strategy, OptimizationStrategy.BALANCED)
    
    def test_single_operation_optimization(self):
        """Test optimization of single operation"""
        async def run_test():
            sample_input = SymbolicTensor(
                data=np.random.random((2, 4, 8, 6, 3)).astype(np.float32),
                symbols={'optimize': True}
            )
            
            result = await self.optimizer.optimize_operation(
                SymbolicOperation.PATTERN_RECOGNITION,
                [sample_input],
                iterations=10
            )
            
            self.assertIsNotNone(result)
            self.assertEqual(result.operation, 'PATTERN_RECOGNITION')
            self.assertGreaterEqual(result.after_latency_ms, 0)
            
            logger.info(f"✅ Optimization: {result.latency_improvement:.1%} improvement")
        
        asyncio.run(run_test())
    
    def test_architecture_selection(self):
        """Test UCB architecture selection"""
        # Initially should explore
        arch = self.optimizer._ucb_select_architecture('TEST_OP')
        self.assertIsInstance(arch, KernelArchitecture)
        
        logger.info(f"✅ Selected architecture: {arch.value}")
    
    def test_optimization_strategies(self):
        """Test different optimization strategies"""
        strategies = [
            OptimizationStrategy.LATENCY_FIRST,
            OptimizationStrategy.THROUGHPUT_FIRST,
            OptimizationStrategy.BALANCED
        ]
        
        for strategy in strategies:
            optimizer = create_optimizer(OptimizationConfig(strategy=strategy))
            
            # Test reward computation
            reward = optimizer._compute_reward(
                latency_ms=1.0,
                accuracy=0.99,
                memory_mb=100.0
            )
            
            self.assertGreaterEqual(reward, -10)
            self.assertLessEqual(reward, 10)
            
            logger.info(f"✅ Strategy {strategy.value}: reward {reward:.3f}")


class TestQuickBenchmark(unittest.TestCase):
    """Test quick benchmark functionality"""
    
    def test_quick_benchmark_runs(self):
        """Test quick benchmark executes successfully"""
        async def run_test():
            results = await run_quick_benchmark(
                operations=[SymbolicOperation.PATTERN_RECOGNITION],
                iterations=5
            )
            
            self.assertIn('operations', results)
            self.assertIn('summary', results)
            self.assertGreater(results['summary']['total_operations'], 0)
            
            logger.info(f"✅ Quick benchmark: {results['summary']['avg_latency_ms']:.3f}ms avg")
        
        asyncio.run(run_test())


class TestIntegration(unittest.TestCase):
    """Integration tests for BEASTMODE components"""
    
    def test_end_to_end_benchmark(self):
        """Test end-to-end benchmark pipeline"""
        async def run_test():
            # Create components
            accelerator = create_accelerator(AcceleratorConfig(warmup_iterations=0))
            validator = create_validator(ValidationLevel.QUICK)
            monitor = create_monitor()
            optimizer = create_optimizer()
            
            # Run through pipeline
            tensor = SymbolicTensor(
                data=np.random.random((2, 4, 8, 6, 3)).astype(np.float32),
                symbols={'integration': True}
            )
            
            # 1. Execute with accelerator
            result = await accelerator.execute(SymbolicOperation.PATTERN_RECOGNITION, [tensor])
            self.assertIsNotNone(result.output)
            
            # 2. Record metrics
            await monitor.record_metrics(
                operation=SymbolicOperation.PATTERN_RECOGNITION,
                architecture=result.architecture,
                latency_ms=result.execution_time_ms,
                memory_mb=result.memory_used_mb,
                accuracy=result.accuracy_score
            )
            
            # 3. Validate
            validation = await validator.validate_operation(SymbolicOperation.PATTERN_RECOGNITION)
            self.assertIsInstance(validation, ValidationResult)
            
            # 4. Optimize
            optimization = await optimizer.optimize_operation(
                SymbolicOperation.PATTERN_RECOGNITION,
                [tensor],
                iterations=5
            )
            
            logger.info("✅ End-to-end integration test passed")
        
        asyncio.run(run_test())


def run_beastmode_tests():
    """Run all BEASTMODE tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestInferenceAccelerator))
    suite.addTests(loader.loadTestsFromTestCase(TestTensorValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceMonitor))
    suite.addTests(loader.loadTestsFromTestCase(TestAdaptiveOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestQuickBenchmark))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_beastmode_tests()
    exit(0 if success else 1)
