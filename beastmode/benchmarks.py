#!/usr/bin/env python3
"""
BEASTMODE Benchmarks
====================

Comprehensive tensor signature benchmarking suite for validation
and performance measurement.
"""

import asyncio
import numpy as np
import time
import logging
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.ggml_symbolic_kernels import (
    SymbolicTensor, SymbolicOperation, KernelArchitecture, get_kernel_manager
)
from src.core.tensor_fragments import TensorShape, Modality

from .inference_engine import InferenceAccelerator, AcceleratorConfig, ExecutionMode
from .tensor_validator import TensorSignatureValidator, ValidationLevel
from .performance_monitor import PerformanceMonitor
from .adaptive_optimizer import AdaptiveOptimizer, OptimizationStrategy, OptimizationConfig

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark suite"""
    tensor_shapes: List[tuple] = None
    operations: List[SymbolicOperation] = None
    iterations_per_test: int = 50
    warmup_iterations: int = 10
    include_validation: bool = True
    include_optimization: bool = True
    export_results: bool = True
    output_dir: str = "/tmp/beastmode_benchmarks"
    
    def __post_init__(self):
        if self.tensor_shapes is None:
            self.tensor_shapes = [
                (2, 4, 8, 6, 3),   # Small
                (4, 8, 16, 8, 4),  # Medium
                (8, 16, 32, 8, 4)  # Large
            ]
        
        if self.operations is None:
            self.operations = [
                SymbolicOperation.PATTERN_RECOGNITION,
                SymbolicOperation.TENSOR_TO_SYMBOL,
                SymbolicOperation.CONTEXT_BINDING,
                SymbolicOperation.SYMBOL_ADD,
                SymbolicOperation.ATTENTION_ROUTING
            ]


async def run_comprehensive_benchmark(config: Optional[BenchmarkConfig] = None) -> Dict[str, Any]:
    """
    Run comprehensive tensor signature benchmarking suite.
    
    Includes:
    - Performance benchmarking across tensor shapes
    - Real-data validation (no mocks)
    - Cross-platform performance analysis
    - Adaptive optimization
    - Regression testing
    - Performance profiling
    
    Returns comprehensive benchmark report.
    """
    config = config or BenchmarkConfig()
    
    logger.info("🚀 BEASTMODE Comprehensive Benchmark Starting")
    logger.info(f"   Shapes: {len(config.tensor_shapes)}, Operations: {len(config.operations)}")
    
    start_time = time.time()
    
    # Initialize components
    accelerator = InferenceAccelerator(AcceleratorConfig(
        enable_profiling=True,
        enable_auto_tuning=True,
        warmup_iterations=config.warmup_iterations
    ))
    
    validator = TensorSignatureValidator(ValidationLevel.STANDARD)
    monitor = PerformanceMonitor()
    optimizer = AdaptiveOptimizer(OptimizationConfig(
        strategy=OptimizationStrategy.BALANCED
    ))
    
    results = {
        'start_time': start_time,
        'config': {
            'tensor_shapes': [str(s) for s in config.tensor_shapes],
            'operations': [op.name for op in config.operations],
            'iterations': config.iterations_per_test
        },
        'performance': {},
        'validation': {},
        'optimization': {},
        'cross_platform': {},
        'summary': {}
    }
    
    # 1. Performance Benchmarking
    logger.info("📊 Phase 1: Performance Benchmarking")
    
    performance_results = []
    
    for shape in config.tensor_shapes:
        shape_key = f"shape_{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}x{shape[4]}"
        results['performance'][shape_key] = {}
        
        for operation in config.operations:
            op_metrics = []
            
            for i in range(config.iterations_per_test):
                # Create test tensor with realistic data
                data = np.random.randn(*shape).astype(np.float32)
                tensor = SymbolicTensor(data=data, symbols={'benchmark': True, 'iteration': i})
                
                # Execute and measure
                exec_result = await accelerator.execute(operation, [tensor])
                
                # Record metrics
                await monitor.record_metrics(
                    operation=operation,
                    architecture=exec_result.architecture,
                    latency_ms=exec_result.execution_time_ms,
                    memory_mb=exec_result.memory_used_mb,
                    accuracy=exec_result.accuracy_score,
                    cache_hit=exec_result.cache_hit
                )
                
                op_metrics.append({
                    'latency_ms': exec_result.execution_time_ms,
                    'accuracy': exec_result.accuracy_score,
                    'cache_hit': exec_result.cache_hit
                })
            
            # Aggregate operation metrics
            latencies = [m['latency_ms'] for m in op_metrics if not m['cache_hit']]
            accuracies = [m['accuracy'] for m in op_metrics]
            
            results['performance'][shape_key][operation.name] = {
                'avg_latency_ms': float(np.mean(latencies)) if latencies else 0.0,
                'p50_latency_ms': float(np.percentile(latencies, 50)) if latencies else 0.0,
                'p95_latency_ms': float(np.percentile(latencies, 95)) if latencies else 0.0,
                'p99_latency_ms': float(np.percentile(latencies, 99)) if len(latencies) >= 100 else 0.0,
                'avg_accuracy': float(np.mean(accuracies)),
                'throughput_ops_sec': 1000.0 / float(np.mean(latencies)) if latencies and np.mean(latencies) > 0 else 0.0,
                'cache_hit_rate': sum(1 for m in op_metrics if m['cache_hit']) / len(op_metrics),
                'iterations': len(op_metrics)
            }
            
            performance_results.append({
                'shape': shape,
                'operation': operation.name,
                **results['performance'][shape_key][operation.name]
            })
    
    # 2. Real-Data Validation
    if config.include_validation:
        logger.info("🧪 Phase 2: Real-Data Validation")
        
        validation_results = await validator.run_comprehensive_validation(
            operations=config.operations
        )
        results['validation'] = validation_results
    
    # 3. Adaptive Optimization
    if config.include_optimization:
        logger.info("⚡ Phase 3: Adaptive Optimization")
        
        optimization_results = await optimizer.run_full_optimization(
            operations=config.operations,
            sample_shape=config.tensor_shapes[0],
            iterations_per_op=20
        )
        results['optimization'] = optimization_results
    
    # 4. Cross-Platform Analysis
    logger.info("🌐 Phase 4: Cross-Platform Analysis")
    
    results['cross_platform'] = _analyze_cross_platform(performance_results)
    
    # 5. Generate Summary
    total_time = time.time() - start_time
    
    results['summary'] = {
        'total_time_seconds': total_time,
        'total_benchmarks': len(performance_results),
        'avg_latency_ms': float(np.mean([r['avg_latency_ms'] for r in performance_results])),
        'avg_accuracy': float(np.mean([r['avg_accuracy'] for r in performance_results])),
        'meets_latency_target': sum(1 for r in performance_results if r['avg_latency_ms'] < 5.0),
        'meets_accuracy_target': sum(1 for r in performance_results if r['avg_accuracy'] > 0.99),
        'cache_efficiency': accelerator.cache.hit_rate if accelerator.cache else 0.0,
        'performance_summary': accelerator.get_performance_summary(),
        'monitor_summary': monitor.get_summary(),
        'validation_pass_rate': results['validation'].get('pass_rate', 0.0) if config.include_validation else None,
        'optimization_improvement': results['optimization'].get('avg_latency_improvement', 0.0) if config.include_optimization else None
    }
    
    logger.info(f"✅ Benchmark Complete: {total_time:.2f}s")
    logger.info(f"   Avg Latency: {results['summary']['avg_latency_ms']:.3f}ms")
    logger.info(f"   Avg Accuracy: {results['summary']['avg_accuracy']:.2%}")
    logger.info(f"   Targets Met: {results['summary']['meets_latency_target']}/{len(performance_results)}")
    
    # Export results
    if config.export_results:
        os.makedirs(config.output_dir, exist_ok=True)
        export_path = os.path.join(config.output_dir, f"benchmark_{int(time.time())}.json")
        with open(export_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"📁 Results exported to {export_path}")
    
    return results


async def run_quick_benchmark(operations: Optional[List[SymbolicOperation]] = None,
                             shape: tuple = (2, 4, 8, 6, 3),
                             iterations: int = 20) -> Dict[str, Any]:
    """
    Run quick benchmark for fast performance assessment.
    
    Useful for CI/CD pipelines and quick validation.
    """
    if operations is None:
        operations = [
            SymbolicOperation.PATTERN_RECOGNITION,
            SymbolicOperation.TENSOR_TO_SYMBOL
        ]
    
    logger.info(f"🏃 Quick Benchmark: {len(operations)} operations, {iterations} iterations")
    
    start_time = time.time()
    kernel_manager = get_kernel_manager()
    
    results = {
        'operations': {},
        'summary': {}
    }
    
    for operation in operations:
        latencies = []
        
        for i in range(iterations):
            data = np.random.randn(*shape).astype(np.float32)
            tensor = SymbolicTensor(data=data, symbols={'quick_bench': True})
            
            start = time.perf_counter()
            await kernel_manager.execute_operation(operation, [tensor])
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)
        
        results['operations'][operation.name] = {
            'avg_latency_ms': float(np.mean(latencies)),
            'p50_latency_ms': float(np.percentile(latencies, 50)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'throughput_ops_sec': 1000.0 / float(np.mean(latencies)),
            'meets_target': np.mean(latencies) < 5.0
        }
    
    total_time = time.time() - start_time
    
    results['summary'] = {
        'total_time_seconds': total_time,
        'total_operations': len(operations),
        'avg_latency_ms': float(np.mean([r['avg_latency_ms'] for r in results['operations'].values()])),
        'all_targets_met': all(r['meets_target'] for r in results['operations'].values())
    }
    
    logger.info(f"✅ Quick Benchmark: {total_time:.2f}s, "
               f"avg {results['summary']['avg_latency_ms']:.3f}ms")
    
    return results


def _analyze_cross_platform(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze cross-platform performance consistency"""
    
    # Group by operation
    by_operation = {}
    for r in results:
        op = r['operation']
        if op not in by_operation:
            by_operation[op] = []
        by_operation[op].append(r['avg_latency_ms'])
    
    # Calculate variance metrics
    analysis = {
        'by_operation': {},
        'overall': {}
    }
    
    for op, latencies in by_operation.items():
        if latencies:
            analysis['by_operation'][op] = {
                'mean_latency_ms': float(np.mean(latencies)),
                'std_latency_ms': float(np.std(latencies)),
                'variance_pct': float(np.std(latencies) / np.mean(latencies) * 100) if np.mean(latencies) > 0 else 0.0,
                'consistent': np.std(latencies) / max(np.mean(latencies), 0.001) < 0.05  # <5% variance
            }
    
    all_latencies = [r['avg_latency_ms'] for r in results]
    analysis['overall'] = {
        'mean_latency_ms': float(np.mean(all_latencies)),
        'std_latency_ms': float(np.std(all_latencies)),
        'variance_pct': float(np.std(all_latencies) / np.mean(all_latencies) * 100) if np.mean(all_latencies) > 0 else 0.0,
        'consistent_operations': sum(1 for v in analysis['by_operation'].values() if v['consistent']),
        'total_operations': len(analysis['by_operation'])
    }
    
    return analysis


def generate_performance_report(benchmark_results: Dict[str, Any]) -> str:
    """Generate human-readable performance report"""
    
    report = []
    report.append("=" * 60)
    report.append("BEASTMODE PERFORMANCE REPORT")
    report.append("=" * 60)
    report.append("")
    
    summary = benchmark_results.get('summary', {})
    
    report.append("📊 SUMMARY")
    report.append("-" * 40)
    report.append(f"Total Time: {summary.get('total_time_seconds', 0):.2f}s")
    report.append(f"Benchmarks Run: {summary.get('total_benchmarks', 0)}")
    report.append(f"Average Latency: {summary.get('avg_latency_ms', 0):.3f}ms")
    report.append(f"Average Accuracy: {summary.get('avg_accuracy', 0):.2%}")
    report.append(f"Latency Target Met: {summary.get('meets_latency_target', 0)} operations")
    report.append(f"Accuracy Target Met: {summary.get('meets_accuracy_target', 0)} operations")
    report.append(f"Cache Hit Rate: {summary.get('cache_efficiency', 0):.1%}")
    report.append("")
    
    if 'validation' in benchmark_results and benchmark_results['validation']:
        report.append("🧪 VALIDATION")
        report.append("-" * 40)
        validation = benchmark_results['validation']
        report.append(f"Pass Rate: {validation.get('pass_rate', 0):.1%}")
        report.append(f"Avg Precision: {validation.get('avg_numerical_precision', 0):.2%}")
        report.append(f"Avg Stability: {validation.get('avg_stability_score', 0):.2%}")
        report.append("")
    
    if 'optimization' in benchmark_results and benchmark_results['optimization']:
        report.append("⚡ OPTIMIZATION")
        report.append("-" * 40)
        optimization = benchmark_results['optimization']
        report.append(f"Successful: {optimization.get('successful_optimizations', 0)}/{optimization.get('total_operations', 0)}")
        report.append(f"Avg Improvement: {optimization.get('avg_latency_improvement', 0):.1%}")
        report.append("")
    
    if 'cross_platform' in benchmark_results:
        report.append("🌐 CROSS-PLATFORM")
        report.append("-" * 40)
        cp = benchmark_results['cross_platform']
        overall = cp.get('overall', {})
        report.append(f"Performance Variance: {overall.get('variance_pct', 0):.1f}%")
        report.append(f"Consistent Operations: {overall.get('consistent_operations', 0)}/{overall.get('total_operations', 0)}")
        report.append("")
    
    report.append("=" * 60)
    
    return "\n".join(report)


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BEASTMODE Benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--full", action="store_true", help="Run comprehensive benchmark")
    parser.add_argument("--iterations", type=int, default=50, help="Iterations per test")
    parser.add_argument("--output", type=str, default="/tmp/beastmode_benchmarks", help="Output directory")
    
    args = parser.parse_args()
    
    if args.quick:
        results = asyncio.run(run_quick_benchmark(iterations=args.iterations))
        print(json.dumps(results, indent=2, default=str))
    elif args.full:
        config = BenchmarkConfig(
            iterations_per_test=args.iterations,
            output_dir=args.output
        )
        results = asyncio.run(run_comprehensive_benchmark(config))
        print(generate_performance_report(results))
    else:
        print("Use --quick or --full to run benchmarks")
