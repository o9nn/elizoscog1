#!/usr/bin/env python3
"""
Enhanced Performance Profiler
Phase 3 Implementation: Comprehensive performance monitoring and profiling

Provides detailed performance metrics, memory usage tracking, cross-platform
analysis, and automated regression testing capabilities.
"""

import asyncio
import numpy as np
import time
import logging
import json
import psutil
import platform
import gc
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.ggml_symbolic_kernels import (
    SymbolicOperation, KernelArchitecture, get_kernel_manager
)

logger = logging.getLogger(__name__)


class ProfilerMetricType(Enum):
    """Types of profiler metrics"""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_UTILIZATION = "cpu_utilization"
    THROUGHPUT = "throughput"
    LATENCY_PERCENTILES = "latency_percentiles"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    SCALABILITY = "scalability"
    REGRESSION = "regression"


@dataclass
class PerformanceSnapshot:
    """Single performance measurement snapshot"""
    timestamp: float
    operation: str
    architecture: str
    
    # Core metrics
    execution_time_ms: float
    memory_before_mb: float
    memory_after_mb: float
    memory_peak_mb: float
    cpu_percent: float
    
    # Derived metrics
    memory_used_mb: float = 0.0
    throughput_ops_per_sec: float = 0.0
    efficiency_score: float = 0.0
    
    # Context information
    input_size_bytes: int = 0
    output_size_bytes: int = 0
    thread_count: int = 1
    system_load: float = 0.0
    
    def __post_init__(self):
        """Calculate derived metrics"""
        self.memory_used_mb = self.memory_after_mb - self.memory_before_mb
        self.throughput_ops_per_sec = 1000.0 / max(self.execution_time_ms, 0.001)
        
        # Efficiency score combining speed and memory efficiency
        speed_score = min(1.0, 10.0 / max(self.execution_time_ms, 0.1))
        memory_score = min(1.0, 100.0 / max(self.memory_used_mb, 0.1))
        self.efficiency_score = 0.7 * speed_score + 0.3 * memory_score


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression detection"""
    operation: str
    architecture: str
    
    baseline_metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    measurement_count: int
    last_updated: float
    
    # Regression thresholds
    latency_regression_threshold: float = 1.2  # 20% increase
    memory_regression_threshold: float = 1.3   # 30% increase
    throughput_regression_threshold: float = 0.8  # 20% decrease


@dataclass 
class RegressionAlert:
    """Alert for detected performance regression"""
    operation: str
    architecture: str
    metric_type: str
    
    baseline_value: float
    current_value: float
    regression_factor: float
    severity: str  # 'minor', 'major', 'critical'
    
    timestamp: float
    additional_context: Dict[str, Any] = field(default_factory=dict)


class EnhancedPerformanceProfiler:
    """Comprehensive performance profiler with regression detection"""
    
    def __init__(self, history_size: int = 1000, baseline_update_interval: int = 100):
        self.kernel_manager = get_kernel_manager()
        
        # Performance data storage
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.performance_baselines: Dict[str, PerformanceBaseline] = {}
        self.regression_alerts: List[RegressionAlert] = []
        
        # Configuration
        self.history_size = history_size
        self.baseline_update_interval = baseline_update_interval
        self.measurement_count = 0
        
        # System monitoring
        self.system_monitor = SystemResourceMonitor()
        
        # Cross-platform tracking
        self.platform_metrics: Dict[str, List[PerformanceSnapshot]] = defaultdict(list)
        
        logger.info(f"Initialized EnhancedPerformanceProfiler with {history_size} history size")
    
    async def profile_operation(self,
                              operation: SymbolicOperation,
                              inputs: List,
                              architecture: KernelArchitecture = KernelArchitecture.CPU_X86_64,
                              metadata: Optional[Dict[str, Any]] = None) -> PerformanceSnapshot:
        """Profile a single operation execution with comprehensive metrics"""
        
        operation_key = f"{operation.name}_{architecture.value}"
        
        # Start system monitoring
        self.system_monitor.start_monitoring()
        
        # Pre-execution measurements
        gc.collect()  # Force garbage collection for accurate memory measurement
        memory_before = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        cpu_before = psutil.cpu_percent(interval=None)
        
        # Calculate input size
        input_size = sum(self._calculate_tensor_size(inp) for inp in inputs)
        
        # Execute operation with timing
        start_time = time.perf_counter()
        
        try:
            result = await self.kernel_manager.execute_operation(operation, inputs, architecture=architecture)
            execution_time = time.perf_counter() - start_time
            
            # Post-execution measurements
            memory_after = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            cpu_after = psutil.cpu_percent(interval=None)
            
            # Get peak memory usage during operation
            peak_memory = self.system_monitor.get_peak_memory_usage()
            
            # Calculate output size
            output_size = self._calculate_tensor_size(result) if result else 0
            
            # Create performance snapshot
            snapshot = PerformanceSnapshot(
                timestamp=time.time(),
                operation=operation.name,
                architecture=architecture.value,
                execution_time_ms=execution_time * 1000,
                memory_before_mb=memory_before,
                memory_after_mb=memory_after,
                memory_peak_mb=peak_memory,
                cpu_percent=(cpu_after - cpu_before) if cpu_after > cpu_before else 0.0,
                input_size_bytes=input_size,
                output_size_bytes=output_size,
                thread_count=threading.active_count(),
                system_load=psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            )
            
            # Store performance data
            self.performance_history[operation_key].append(snapshot)
            self.platform_metrics[architecture.value].append(snapshot)
            self.measurement_count += 1
            
            # Update baselines periodically
            if self.measurement_count % self.baseline_update_interval == 0:
                await self._update_baselines(operation_key)
            
            # Check for regressions
            await self._check_for_regressions(operation_key, snapshot)
            
            logger.debug(f"Profiled {operation.name} on {architecture.value}: "
                        f"{snapshot.execution_time_ms:.3f}ms, "
                        f"{snapshot.memory_used_mb:.2f}MB")
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to profile {operation.name}: {e}")
            # Create error snapshot
            return PerformanceSnapshot(
                timestamp=time.time(),
                operation=operation.name,
                architecture=architecture.value,
                execution_time_ms=999999.0,  # High penalty for failure
                memory_before_mb=memory_before,
                memory_after_mb=memory_before,
                memory_peak_mb=memory_before,
                cpu_percent=0.0
            )
        
        finally:
            self.system_monitor.stop_monitoring()
    
    def _calculate_tensor_size(self, tensor) -> int:
        """Calculate size of tensor in bytes"""
        try:
            if hasattr(tensor, 'data') and hasattr(tensor.data, 'nbytes'):
                return tensor.data.nbytes
            elif hasattr(tensor, 'size'):
                return tensor.size * 4  # Assume float32
            else:
                return 0
        except:
            return 0
    
    async def _update_baselines(self, operation_key: str):
        """Update performance baselines for regression detection"""
        if operation_key not in self.performance_history:
            return
        
        history = list(self.performance_history[operation_key])
        if len(history) < 10:  # Need minimum data for baseline
            return
        
        # Calculate baseline metrics from recent performance data
        recent_data = history[-50:]  # Use last 50 measurements
        
        latencies = [s.execution_time_ms for s in recent_data]
        memory_usage = [s.memory_used_mb for s in recent_data]
        throughput = [s.throughput_ops_per_sec for s in recent_data]
        
        baseline_metrics = {
            'latency_ms': np.mean(latencies),
            'memory_mb': np.mean(memory_usage),
            'throughput_ops_sec': np.mean(throughput)
        }
        
        # Calculate confidence intervals (95%)
        confidence_intervals = {
            'latency_ms': (np.percentile(latencies, 2.5), np.percentile(latencies, 97.5)),
            'memory_mb': (np.percentile(memory_usage, 2.5), np.percentile(memory_usage, 97.5)),
            'throughput_ops_sec': (np.percentile(throughput, 2.5), np.percentile(throughput, 97.5))
        }
        
        # Extract operation and architecture from key
        parts = operation_key.split('_', 1)
        operation = parts[0] if len(parts) > 0 else 'unknown'
        architecture = parts[1] if len(parts) > 1 else 'unknown'
        
        # Create or update baseline
        self.performance_baselines[operation_key] = PerformanceBaseline(
            operation=operation,
            architecture=architecture,
            baseline_metrics=baseline_metrics,
            confidence_intervals=confidence_intervals,
            measurement_count=len(recent_data),
            last_updated=time.time()
        )
        
        logger.info(f"Updated baseline for {operation_key}: "
                   f"{baseline_metrics['latency_ms']:.3f}ms latency, "
                   f"{baseline_metrics['throughput_ops_sec']:.1f} ops/sec")
    
    async def _check_for_regressions(self, operation_key: str, snapshot: PerformanceSnapshot):
        """Check for performance regressions against baseline"""
        if operation_key not in self.performance_baselines:
            return
        
        baseline = self.performance_baselines[operation_key]
        
        # Check latency regression
        baseline_latency = baseline.baseline_metrics['latency_ms']
        if snapshot.execution_time_ms > baseline_latency * baseline.latency_regression_threshold:
            regression_factor = snapshot.execution_time_ms / baseline_latency
            severity = self._determine_regression_severity(regression_factor, 'latency')
            
            alert = RegressionAlert(
                operation=baseline.operation,
                architecture=baseline.architecture,
                metric_type='latency',
                baseline_value=baseline_latency,
                current_value=snapshot.execution_time_ms,
                regression_factor=regression_factor,
                severity=severity,
                timestamp=snapshot.timestamp,
                additional_context={
                    'baseline_confidence_interval': baseline.confidence_intervals['latency_ms'],
                    'system_load': snapshot.system_load,
                    'memory_usage': snapshot.memory_used_mb
                }
            )
            
            self.regression_alerts.append(alert)
            logger.warning(f"REGRESSION DETECTED: {operation_key} latency increased by "
                          f"{(regression_factor - 1) * 100:.1f}% (severity: {severity})")
        
        # Check memory regression
        baseline_memory = baseline.baseline_metrics['memory_mb']
        if snapshot.memory_used_mb > baseline_memory * baseline.memory_regression_threshold:
            regression_factor = snapshot.memory_used_mb / max(baseline_memory, 0.1)
            severity = self._determine_regression_severity(regression_factor, 'memory')
            
            alert = RegressionAlert(
                operation=baseline.operation,
                architecture=baseline.architecture,
                metric_type='memory',
                baseline_value=baseline_memory,
                current_value=snapshot.memory_used_mb,
                regression_factor=regression_factor,
                severity=severity,
                timestamp=snapshot.timestamp
            )
            
            self.regression_alerts.append(alert)
            logger.warning(f"REGRESSION DETECTED: {operation_key} memory usage increased by "
                          f"{(regression_factor - 1) * 100:.1f}% (severity: {severity})")
        
        # Check throughput regression
        baseline_throughput = baseline.baseline_metrics['throughput_ops_sec']
        if snapshot.throughput_ops_per_sec < baseline_throughput * baseline.throughput_regression_threshold:
            regression_factor = baseline_throughput / max(snapshot.throughput_ops_per_sec, 0.1)
            severity = self._determine_regression_severity(regression_factor, 'throughput')
            
            alert = RegressionAlert(
                operation=baseline.operation,
                architecture=baseline.architecture,
                metric_type='throughput',
                baseline_value=baseline_throughput,
                current_value=snapshot.throughput_ops_per_sec,
                regression_factor=regression_factor,
                severity=severity,
                timestamp=snapshot.timestamp
            )
            
            self.regression_alerts.append(alert)
            logger.warning(f"REGRESSION DETECTED: {operation_key} throughput decreased by "
                          f"{(1 - 1/regression_factor) * 100:.1f}% (severity: {severity})")
    
    def _determine_regression_severity(self, regression_factor: float, metric_type: str) -> str:
        """Determine severity of performance regression"""
        if metric_type == 'latency':
            if regression_factor > 2.0:
                return 'critical'
            elif regression_factor > 1.5:
                return 'major'
            else:
                return 'minor'
        elif metric_type == 'memory':
            if regression_factor > 3.0:
                return 'critical'
            elif regression_factor > 2.0:
                return 'major'
            else:
                return 'minor'
        elif metric_type == 'throughput':
            if regression_factor > 2.0:
                return 'critical'
            elif regression_factor > 1.5:
                return 'major'
            else:
                return 'minor'
        
        return 'minor'
    
    def get_performance_summary(self, 
                              operation: Optional[str] = None,
                              architecture: Optional[str] = None,
                              time_window_hours: float = 24.0) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        cutoff_time = time.time() - (time_window_hours * 3600)
        
        summary = {
            'time_window_hours': time_window_hours,
            'total_measurements': 0,
            'operations_analyzed': set(),
            'architectures_analyzed': set(),
            'performance_metrics': {},
            'cross_platform_analysis': {},
            'regression_summary': {}
        }
        
        # Collect relevant measurements
        relevant_measurements = []
        
        for operation_key, history in self.performance_history.items():
            op_name, arch_name = operation_key.split('_', 1)
            
            if operation and op_name != operation:
                continue
            if architecture and arch_name != architecture:
                continue
            
            recent_measurements = [s for s in history if s.timestamp > cutoff_time]
            relevant_measurements.extend(recent_measurements)
            
            if recent_measurements:
                summary['operations_analyzed'].add(op_name)
                summary['architectures_analyzed'].add(arch_name)
        
        summary['total_measurements'] = len(relevant_measurements)
        summary['operations_analyzed'] = list(summary['operations_analyzed'])
        summary['architectures_analyzed'] = list(summary['architectures_analyzed'])
        
        if not relevant_measurements:
            return summary
        
        # Calculate aggregate performance metrics
        latencies = [s.execution_time_ms for s in relevant_measurements]
        memory_usage = [s.memory_used_mb for s in relevant_measurements]
        throughputs = [s.throughput_ops_per_sec for s in relevant_measurements]
        efficiency_scores = [s.efficiency_score for s in relevant_measurements]
        
        summary['performance_metrics'] = {
            'latency_stats': {
                'mean_ms': np.mean(latencies),
                'median_ms': np.median(latencies),
                'p95_ms': np.percentile(latencies, 95),
                'p99_ms': np.percentile(latencies, 99),
                'std_ms': np.std(latencies),
                'meets_5ms_target': np.mean([l < 5.0 for l in latencies])
            },
            'memory_stats': {
                'mean_mb': np.mean(memory_usage),
                'median_mb': np.median(memory_usage),
                'p95_mb': np.percentile(memory_usage, 95),
                'max_mb': np.max(memory_usage),
                'std_mb': np.std(memory_usage)
            },
            'throughput_stats': {
                'mean_ops_per_sec': np.mean(throughputs),
                'median_ops_per_sec': np.median(throughputs),
                'p95_ops_per_sec': np.percentile(throughputs, 95),
                'std_ops_per_sec': np.std(throughputs)
            },
            'efficiency_stats': {
                'mean_score': np.mean(efficiency_scores),
                'median_score': np.median(efficiency_scores),
                'high_efficiency_ratio': np.mean([e > 0.8 for e in efficiency_scores])
            }
        }
        
        # Cross-platform analysis
        if len(summary['architectures_analyzed']) > 1:
            summary['cross_platform_analysis'] = self._analyze_cross_platform_performance(
                relevant_measurements
            )
        
        # Regression analysis
        recent_alerts = [alert for alert in self.regression_alerts 
                        if alert.timestamp > cutoff_time]
        
        summary['regression_summary'] = {
            'total_alerts': len(recent_alerts),
            'critical_alerts': len([a for a in recent_alerts if a.severity == 'critical']),
            'major_alerts': len([a for a in recent_alerts if a.severity == 'major']),
            'minor_alerts': len([a for a in recent_alerts if a.severity == 'minor']),
            'most_common_regression_type': self._get_most_common_regression_type(recent_alerts),
            'alert_details': [
                {
                    'operation': alert.operation,
                    'architecture': alert.architecture,
                    'metric_type': alert.metric_type,
                    'regression_factor': alert.regression_factor,
                    'severity': alert.severity
                }
                for alert in recent_alerts[-10:]  # Last 10 alerts
            ]
        }
        
        return summary
    
    def _analyze_cross_platform_performance(self, measurements: List[PerformanceSnapshot]) -> Dict[str, Any]:
        """Analyze performance consistency across platforms"""
        
        platform_groups = defaultdict(list)
        for measurement in measurements:
            platform_groups[measurement.architecture].append(measurement)
        
        analysis = {
            'platform_comparison': {},
            'consistency_metrics': {},
            'recommendations': []
        }
        
        # Compare performance across platforms
        for platform, platform_measurements in platform_groups.items():
            if len(platform_measurements) < 5:  # Need minimum data
                continue
            
            latencies = [m.execution_time_ms for m in platform_measurements]
            memory_usage = [m.memory_used_mb for m in platform_measurements]
            throughputs = [m.throughput_ops_per_sec for m in platform_measurements]
            
            analysis['platform_comparison'][platform] = {
                'measurement_count': len(platform_measurements),
                'avg_latency_ms': np.mean(latencies),
                'avg_memory_mb': np.mean(memory_usage),
                'avg_throughput_ops_sec': np.mean(throughputs),
                'latency_consistency': 1.0 - np.std(latencies) / max(np.mean(latencies), 0.001),
                'relative_performance': 1.0  # Will be calculated below
            }
        
        # Calculate relative performance scores
        if len(analysis['platform_comparison']) > 1:
            # Use fastest platform as reference
            reference_latency = min(
                data['avg_latency_ms'] 
                for data in analysis['platform_comparison'].values()
            )
            
            for platform, data in analysis['platform_comparison'].items():
                data['relative_performance'] = reference_latency / data['avg_latency_ms']
        
        # Calculate overall consistency metrics
        if len(platform_groups) > 1:
            all_latencies = []
            platform_avg_latencies = []
            
            for platform_measurements in platform_groups.values():
                platform_latencies = [m.execution_time_ms for m in platform_measurements]
                all_latencies.extend(platform_latencies)
                platform_avg_latencies.append(np.mean(platform_latencies))
            
            # Cross-platform variance
            cross_platform_variance = np.std(platform_avg_latencies) / max(np.mean(platform_avg_latencies), 0.001)
            
            analysis['consistency_metrics'] = {
                'cross_platform_variance': cross_platform_variance,
                'meets_5_percent_target': cross_platform_variance < 0.05,
                'performance_spread': (max(platform_avg_latencies) - min(platform_avg_latencies)) / min(platform_avg_latencies)
            }
            
            # Generate recommendations
            if cross_platform_variance > 0.1:
                analysis['recommendations'].append("High cross-platform variance detected - investigate platform-specific optimizations")
            
            worst_platform = max(analysis['platform_comparison'].items(), 
                               key=lambda x: x[1]['avg_latency_ms'])
            if worst_platform[1]['relative_performance'] < 0.8:
                analysis['recommendations'].append(f"Platform {worst_platform[0]} shows poor performance - consider architecture-specific optimizations")
        
        return analysis
    
    def _get_most_common_regression_type(self, alerts: List[RegressionAlert]) -> str:
        """Get the most common type of regression"""
        if not alerts:
            return 'none'
        
        type_counts = defaultdict(int)
        for alert in alerts:
            type_counts[alert.metric_type] += 1
        
        return max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else 'none'
    
    async def run_performance_benchmark(self,
                                      operations: List[SymbolicOperation],
                                      test_data_generator: Callable,
                                      architectures: Optional[List[KernelArchitecture]] = None,
                                      iterations: int = 50) -> Dict[str, Any]:
        """Run comprehensive performance benchmark"""
        
        if architectures is None:
            architectures = self.kernel_manager.get_available_architectures()
        
        logger.info(f"Running performance benchmark: {len(operations)} operations, "
                   f"{len(architectures)} architectures, {iterations} iterations")
        
        benchmark_results = {
            'benchmark_config': {
                'operations': [op.name for op in operations],
                'architectures': [arch.value for arch in architectures],
                'iterations': iterations,
                'start_time': time.time()
            },
            'results': {},
            'performance_summary': {},
            'regression_analysis': {}
        }
        
        # Run benchmarks
        for operation in operations:
            for architecture in architectures:
                operation_key = f"{operation.name}_{architecture.value}"
                snapshots = []
                
                logger.info(f"Benchmarking {operation_key}...")
                
                for i in range(iterations):
                    try:
                        # Generate test data
                        test_inputs = test_data_generator()
                        
                        # Profile operation
                        snapshot = await self.profile_operation(operation, test_inputs, architecture)
                        snapshots.append(snapshot)
                        
                    except Exception as e:
                        logger.warning(f"Benchmark iteration {i} failed for {operation_key}: {e}")
                
                if snapshots:
                    # Calculate benchmark statistics
                    latencies = [s.execution_time_ms for s in snapshots]
                    memory_usage = [s.memory_used_mb for s in snapshots]
                    throughputs = [s.throughput_ops_per_sec for s in snapshots]
                    
                    benchmark_results['results'][operation_key] = {
                        'successful_iterations': len(snapshots),
                        'latency_stats': {
                            'mean_ms': np.mean(latencies),
                            'median_ms': np.median(latencies),
                            'p95_ms': np.percentile(latencies, 95),
                            'p99_ms': np.percentile(latencies, 99),
                            'std_ms': np.std(latencies),
                            'min_ms': np.min(latencies),
                            'max_ms': np.max(latencies)
                        },
                        'memory_stats': {
                            'mean_mb': np.mean(memory_usage),
                            'median_mb': np.median(memory_usage),
                            'p95_mb': np.percentile(memory_usage, 95),
                            'max_mb': np.max(memory_usage),
                            'std_mb': np.std(memory_usage)
                        },
                        'throughput_stats': {
                            'mean_ops_per_sec': np.mean(throughputs),
                            'median_ops_per_sec': np.median(throughputs),
                            'p95_ops_per_sec': np.percentile(throughputs, 95),
                            'std_ops_per_sec': np.std(throughputs)
                        },
                        'performance_targets': {
                            'meets_5ms_latency': np.mean(latencies) < 5.0,
                            'meets_consistency': np.std(latencies) / np.mean(latencies) < 0.2,
                            'meets_memory_efficiency': np.mean(memory_usage) < 100.0
                        }
                    }
        
        benchmark_results['benchmark_config']['end_time'] = time.time()
        benchmark_results['benchmark_config']['total_duration_seconds'] = (
            benchmark_results['benchmark_config']['end_time'] - 
            benchmark_results['benchmark_config']['start_time']
        )
        
        # Generate performance summary
        benchmark_results['performance_summary'] = self._generate_benchmark_summary(benchmark_results['results'])
        
        # Generate regression analysis
        benchmark_results['regression_analysis'] = self._analyze_benchmark_regressions()
        
        logger.info(f"Performance benchmark complete in "
                   f"{benchmark_results['benchmark_config']['total_duration_seconds']:.1f}s")
        
        return benchmark_results
    
    def _generate_benchmark_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of benchmark results"""
        if not results:
            return {}
        
        all_latencies = []
        all_memory = []
        all_throughputs = []
        target_compliance = []
        
        for operation_result in results.values():
            all_latencies.append(operation_result['latency_stats']['mean_ms'])
            all_memory.append(operation_result['memory_stats']['mean_mb'])
            all_throughputs.append(operation_result['throughput_stats']['mean_ops_per_sec'])
            
            targets = operation_result['performance_targets']
            compliance = sum([
                targets['meets_5ms_latency'],
                targets['meets_consistency'],
                targets['meets_memory_efficiency']
            ]) / 3.0
            target_compliance.append(compliance)
        
        return {
            'aggregate_performance': {
                'avg_latency_ms': np.mean(all_latencies),
                'avg_memory_mb': np.mean(all_memory),
                'avg_throughput_ops_sec': np.mean(all_throughputs),
                'overall_target_compliance': np.mean(target_compliance)
            },
            'performance_distribution': {
                'latency_p50_ms': np.percentile(all_latencies, 50),
                'latency_p95_ms': np.percentile(all_latencies, 95),
                'memory_p95_mb': np.percentile(all_memory, 95),
                'throughput_p50_ops_sec': np.percentile(all_throughputs, 50)
            },
            'target_achievement': {
                'operations_meeting_latency_target': sum(1 for l in all_latencies if l < 5.0) / len(all_latencies),
                'operations_meeting_memory_target': sum(1 for m in all_memory if m < 100.0) / len(all_memory),
                'overall_success_rate': np.mean(target_compliance)
            }
        }
    
    def _analyze_benchmark_regressions(self) -> Dict[str, Any]:
        """Analyze regressions detected during benchmarking"""
        recent_alerts = self.regression_alerts[-50:]  # Last 50 alerts
        
        if not recent_alerts:
            return {'status': 'no_regressions_detected'}
        
        # Group alerts by type and severity
        by_type = defaultdict(list)
        by_severity = defaultdict(list)
        by_operation = defaultdict(list)
        
        for alert in recent_alerts:
            by_type[alert.metric_type].append(alert)
            by_severity[alert.severity].append(alert)
            by_operation[alert.operation].append(alert)
        
        return {
            'total_regressions': len(recent_alerts),
            'by_type': {
                metric_type: {
                    'count': len(alerts),
                    'avg_regression_factor': np.mean([a.regression_factor for a in alerts])
                }
                for metric_type, alerts in by_type.items()
            },
            'by_severity': {
                severity: len(alerts)
                for severity, alerts in by_severity.items()
            },
            'most_problematic_operations': [
                {
                    'operation': operation,
                    'regression_count': len(alerts),
                    'avg_severity_score': self._calculate_severity_score(alerts)
                }
                for operation, alerts in sorted(by_operation.items(), 
                                              key=lambda x: len(x[1]), reverse=True)[:5]
            ],
            'recommendation': self._generate_regression_recommendations(recent_alerts)
        }
    
    def _calculate_severity_score(self, alerts: List[RegressionAlert]) -> float:
        """Calculate average severity score for alerts"""
        severity_map = {'minor': 1, 'major': 2, 'critical': 3}
        return np.mean([severity_map.get(alert.severity, 1) for alert in alerts])
    
    def _generate_regression_recommendations(self, alerts: List[RegressionAlert]) -> List[str]:
        """Generate recommendations based on regression analysis"""
        recommendations = []
        
        # Analyze patterns in alerts
        latency_alerts = [a for a in alerts if a.metric_type == 'latency']
        memory_alerts = [a for a in alerts if a.metric_type == 'memory']
        throughput_alerts = [a for a in alerts if a.metric_type == 'throughput']
        
        if len(latency_alerts) > len(alerts) * 0.5:
            recommendations.append("High number of latency regressions detected - focus on performance optimization")
        
        if len(memory_alerts) > len(alerts) * 0.3:
            recommendations.append("Memory usage regressions detected - review memory management and garbage collection")
        
        if len(throughput_alerts) > len(alerts) * 0.2:
            recommendations.append("Throughput regressions detected - investigate bottlenecks and resource contention")
        
        # Check for critical alerts
        critical_alerts = [a for a in alerts if a.severity == 'critical']
        if len(critical_alerts) > 0:
            recommendations.append(f"URGENT: {len(critical_alerts)} critical performance regressions require immediate attention")
        
        if not recommendations:
            recommendations.append("Performance appears stable - continue monitoring")
        
        return recommendations
    
    def get_regression_report(self, time_window_hours: float = 24.0) -> Dict[str, Any]:
        """Get comprehensive regression analysis report"""
        cutoff_time = time.time() - (time_window_hours * 3600)
        recent_alerts = [alert for alert in self.regression_alerts if alert.timestamp > cutoff_time]
        
        report = {
            'time_window_hours': time_window_hours,
            'total_alerts': len(recent_alerts),
            'alert_timeline': [],
            'severity_breakdown': {},
            'operation_analysis': {},
            'architecture_analysis': {},
            'trend_analysis': {}
        }
        
        if not recent_alerts:
            report['status'] = 'no_regressions_in_window'
            return report
        
        # Timeline analysis
        sorted_alerts = sorted(recent_alerts, key=lambda x: x.timestamp)
        for alert in sorted_alerts[-20:]:  # Last 20 alerts
            report['alert_timeline'].append({
                'timestamp': alert.timestamp,
                'operation': alert.operation,
                'architecture': alert.architecture,
                'metric_type': alert.metric_type,
                'regression_factor': alert.regression_factor,
                'severity': alert.severity
            })
        
        # Severity breakdown
        by_severity = defaultdict(int)
        for alert in recent_alerts:
            by_severity[alert.severity] += 1
        report['severity_breakdown'] = dict(by_severity)
        
        # Operation analysis
        by_operation = defaultdict(list)
        for alert in recent_alerts:
            by_operation[alert.operation].append(alert)
        
        for operation, operation_alerts in by_operation.items():
            report['operation_analysis'][operation] = {
                'total_alerts': len(operation_alerts),
                'severity_distribution': {
                    severity: len([a for a in operation_alerts if a.severity == severity])
                    for severity in ['minor', 'major', 'critical']
                },
                'avg_regression_factor': np.mean([a.regression_factor for a in operation_alerts]),
                'most_common_metric': max(
                    {metric: len([a for a in operation_alerts if a.metric_type == metric])
                     for metric in ['latency', 'memory', 'throughput']}.items(),
                    key=lambda x: x[1]
                )[0] if operation_alerts else 'none'
            }
        
        # Architecture analysis  
        by_architecture = defaultdict(list)
        for alert in recent_alerts:
            by_architecture[alert.architecture].append(alert)
        
        for architecture, arch_alerts in by_architecture.items():
            report['architecture_analysis'][architecture] = {
                'total_alerts': len(arch_alerts),
                'avg_regression_factor': np.mean([a.regression_factor for a in arch_alerts]),
                'problem_operations': list(set(a.operation for a in arch_alerts))
            }
        
        return report
    
    def export_performance_data(self, filepath: str, include_raw_data: bool = False) -> None:
        """Export performance profiling data to JSON file"""
        
        export_data = {
            'export_timestamp': time.time(),
            'system_info': {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
                'total_memory_gb': psutil.virtual_memory().total / (1024**3)
            },
            'profiler_config': {
                'history_size': self.history_size,
                'baseline_update_interval': self.baseline_update_interval,
                'total_measurements': self.measurement_count
            },
            'performance_summary': self.get_performance_summary(),
            'baseline_data': {},
            'regression_alerts': []
        }
        
        # Export baselines
        for key, baseline in self.performance_baselines.items():
            export_data['baseline_data'][key] = {
                'operation': baseline.operation,
                'architecture': baseline.architecture,
                'baseline_metrics': baseline.baseline_metrics,
                'confidence_intervals': baseline.confidence_intervals,
                'measurement_count': baseline.measurement_count,
                'last_updated': baseline.last_updated
            }
        
        # Export recent regression alerts
        recent_alerts = self.regression_alerts[-100:]  # Last 100 alerts
        for alert in recent_alerts:
            export_data['regression_alerts'].append({
                'operation': alert.operation,
                'architecture': alert.architecture,
                'metric_type': alert.metric_type,
                'baseline_value': alert.baseline_value,
                'current_value': alert.current_value,
                'regression_factor': alert.regression_factor,
                'severity': alert.severity,
                'timestamp': alert.timestamp
            })
        
        # Optionally include raw performance data
        if include_raw_data:
            export_data['raw_performance_data'] = {}
            for key, history in self.performance_history.items():
                export_data['raw_performance_data'][key] = [
                    {
                        'timestamp': snapshot.timestamp,
                        'execution_time_ms': snapshot.execution_time_ms,
                        'memory_used_mb': snapshot.memory_used_mb,
                        'throughput_ops_per_sec': snapshot.throughput_ops_per_sec,
                        'efficiency_score': snapshot.efficiency_score,
                        'input_size_bytes': snapshot.input_size_bytes,
                        'output_size_bytes': snapshot.output_size_bytes
                    }
                    for snapshot in list(history)[-100:]  # Last 100 measurements per operation
                ]
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Performance data exported to {filepath}")


class SystemResourceMonitor:
    """Monitor system resources during operation execution"""
    
    def __init__(self):
        self.monitoring = False
        self.peak_memory_mb = 0.0
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start monitoring system resources"""
        self.monitoring = True
        self.peak_memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring system resources"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def get_peak_memory_usage(self) -> float:
        """Get peak memory usage during monitoring period"""
        return self.peak_memory_mb
    
    def _monitor_resources(self):
        """Background thread to monitor resources"""
        while self.monitoring:
            try:
                current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                self.peak_memory_mb = max(self.peak_memory_mb, current_memory)
                time.sleep(0.01)  # Monitor every 10ms
            except:
                break


# Integration with Adaptive Optimization
class AdaptivePerformanceProfiler(EnhancedPerformanceProfiler):
    """Enhanced performance profiler with adaptive optimization integration"""
    
    def __init__(self, history_size: int = 1000, enable_adaptive_optimization: bool = True):
        super().__init__(history_size)
        self.enable_adaptive_optimization = enable_adaptive_optimization
        self.adaptive_engine = None
        self.optimization_callbacks = []
        
        # Adaptive optimization metrics
        self.adaptive_improvement_history = []
        self.last_optimization_trigger = 0.0
        self.optimization_cooldown = 300.0  # 5 minutes
        
    def register_adaptive_engine(self, adaptive_engine):
        """Register an adaptive optimization engine for integration"""
        self.adaptive_engine = adaptive_engine
        logger.info("🔗 Adaptive optimization engine registered with performance profiler")
    
    def add_optimization_callback(self, callback: Callable):
        """Add callback to be triggered on performance issues"""
        self.optimization_callbacks.append(callback)
    
    async def trigger_adaptive_optimization(self, regression_alert: RegressionAlert) -> Dict[str, Any]:
        """Trigger adaptive optimization in response to performance regression"""
        if not self.enable_adaptive_optimization:
            return {"status": "adaptive_optimization_disabled"}
        
        # Check cooldown period
        if time.time() - self.last_optimization_trigger < self.optimization_cooldown:
            return {"status": "cooldown_active", "remaining_seconds": 
                   self.optimization_cooldown - (time.time() - self.last_optimization_trigger)}
        
        self.last_optimization_trigger = time.time()
        
        # Prepare optimization context
        optimization_context = {
            "trigger_type": "performance_regression",
            "regression_details": {
                "operation": regression_alert.operation,
                "architecture": regression_alert.architecture,
                "metric_type": regression_alert.metric_type,
                "regression_factor": regression_alert.regression_factor,
                "severity": regression_alert.severity
            },
            "current_baselines": self.performance_baselines.copy(),
            "recent_performance": self.get_performance_summary()
        }
        
        # Trigger adaptive engine if available
        optimization_result = {"status": "no_adaptive_engine"}
        if self.adaptive_engine:
            try:
                # This would integrate with the adaptive optimization engine
                optimization_result = await self._execute_adaptive_optimization(optimization_context)
            except Exception as e:
                logger.error(f"Adaptive optimization failed: {e}")
                optimization_result = {"status": "optimization_failed", "error": str(e)}
        
        # Execute callbacks
        for callback in self.optimization_callbacks:
            try:
                await callback(optimization_context, optimization_result)
            except Exception as e:
                logger.error(f"Optimization callback failed: {e}")
        
        # Record improvement attempt
        self.adaptive_improvement_history.append({
            "timestamp": time.time(),
            "trigger": regression_alert,
            "optimization_context": optimization_context,
            "result": optimization_result
        })
        
        logger.info(f"🔧 Adaptive optimization triggered for {regression_alert.operation} "
                   f"({regression_alert.severity} {regression_alert.metric_type} regression)")
        
        return optimization_result
    
    async def _execute_adaptive_optimization(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute adaptive optimization with the registered engine"""
        if not self.adaptive_engine:
            return {"status": "no_engine"}
        
        # Get current engine status
        engine_status = self.adaptive_engine.get_comprehensive_status()
        
        # Check if engine is already running optimization
        if engine_status["engine_status"]["running"]:
            return {"status": "optimization_already_running"}
        
        # Create adaptive parameters based on regression type
        regression_details = context["regression_details"]
        suggested_parameters = self._suggest_optimization_parameters(regression_details)
        
        # This would typically trigger parameter adjustment or re-optimization
        optimization_suggestion = {
            "parameter_adjustments": suggested_parameters,
            "optimization_strategy": self._recommend_optimization_strategy(regression_details),
            "expected_improvement": self._estimate_improvement_potential(regression_details)
        }
        
        return {
            "status": "optimization_suggested",
            "suggestions": optimization_suggestion,
            "engine_status": engine_status
        }
    
    def _suggest_optimization_parameters(self, regression_details: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest parameter adjustments based on regression type"""
        metric_type = regression_details.get("metric_type", "unknown")
        severity = regression_details.get("severity", "minor")
        
        suggestions = {}
        
        if metric_type == "latency":
            suggestions.update({
                "execution_optimization": "increase",
                "parallelization": "consider",
                "caching_strategy": "aggressive"
            })
        elif metric_type == "memory":
            suggestions.update({
                "memory_management": "optimize",
                "garbage_collection": "tune",
                "buffer_sizes": "reduce"
            })
        elif metric_type == "throughput":
            suggestions.update({
                "batch_processing": "optimize",
                "resource_allocation": "increase",
                "bottleneck_analysis": "required"
            })
        
        # Adjust aggressiveness based on severity
        if severity == "critical":
            suggestions["optimization_aggressiveness"] = "high"
            suggestions["immediate_action"] = "required"
        elif severity == "major":
            suggestions["optimization_aggressiveness"] = "medium"
        else:
            suggestions["optimization_aggressiveness"] = "conservative"
        
        return suggestions
    
    def _recommend_optimization_strategy(self, regression_details: Dict[str, Any]) -> str:
        """Recommend optimization strategy based on regression pattern"""
        metric_type = regression_details.get("metric_type", "unknown")
        severity = regression_details.get("severity", "minor")
        
        if severity == "critical":
            return "emergency_optimization"
        elif metric_type in ["latency", "throughput"]:
            return "performance_focused"
        elif metric_type == "memory":
            return "resource_efficiency"
        else:
            return "balanced_optimization"
    
    def _estimate_improvement_potential(self, regression_details: Dict[str, Any]) -> float:
        """Estimate potential improvement from optimization"""
        regression_factor = regression_details.get("regression_factor", 1.0)
        severity = regression_details.get("severity", "minor")
        
        # Base improvement potential based on regression severity
        base_potential = min(0.5, (regression_factor - 1.0) * 0.8)
        
        # Adjust based on severity
        severity_multiplier = {"minor": 0.7, "major": 0.9, "critical": 1.2}.get(severity, 0.8)
        
        return min(1.0, base_potential * severity_multiplier)
    
    def get_adaptive_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of adaptive optimization activities"""
        if not self.adaptive_improvement_history:
            return {"status": "no_optimization_history"}
        
        recent_optimizations = self.adaptive_improvement_history[-10:]
        
        # Calculate success metrics
        successful_optimizations = [
            opt for opt in recent_optimizations 
            if opt["result"].get("status") not in ["optimization_failed", "no_engine"]
        ]
        
        success_rate = len(successful_optimizations) / len(recent_optimizations) if recent_optimizations else 0.0
        
        # Analyze optimization triggers
        trigger_analysis = {}
        for opt in recent_optimizations:
            trigger = opt["trigger"]
            key = f"{trigger.metric_type}_{trigger.severity}"
            trigger_analysis[key] = trigger_analysis.get(key, 0) + 1
        
        return {
            "total_optimization_attempts": len(self.adaptive_improvement_history),
            "recent_attempts": len(recent_optimizations),
            "success_rate": success_rate,
            "trigger_analysis": trigger_analysis,
            "last_optimization": self.adaptive_improvement_history[-1]["timestamp"] if self.adaptive_improvement_history else None,
            "cooldown_remaining": max(0, self.optimization_cooldown - (time.time() - self.last_optimization_trigger)),
            "adaptive_engine_available": self.adaptive_engine is not None
        }


# Factory function for easy instantiation
def create_performance_profiler(history_size: int = 1000, baseline_update_interval: int = 100) -> EnhancedPerformanceProfiler:
    """Create and return a configured performance profiler"""
    return EnhancedPerformanceProfiler(history_size=history_size, baseline_update_interval=baseline_update_interval)


def create_adaptive_performance_profiler(history_size: int = 1000, 
                                       enable_adaptive_optimization: bool = True) -> AdaptivePerformanceProfiler:
    """Create and return an adaptive performance profiler with optimization capabilities"""
    return AdaptivePerformanceProfiler(history_size=history_size, 
                                     enable_adaptive_optimization=enable_adaptive_optimization)