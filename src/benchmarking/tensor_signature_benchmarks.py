#!/usr/bin/env python3
"""
Tensor Signature Benchmarking Suite
Phase 3 Implementation: Comprehensive benchmarks for diverse tensor signatures and operations

Implements comprehensive benchmarking for all tensor shapes, operations, and architectures
with real-world performance validation and optimization recommendations.
"""

import asyncio
import numpy as np
import time
import logging
import json
import platform
import psutil
import gc
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import statistics
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.ggml_symbolic_kernels import (
    SymbolicTensor, GGMLSymbolicKernelManager, SymbolicOperation,
    KernelArchitecture, get_kernel_manager
)
from src.core.tensor_fragments import (
    TensorShape, TensorFragment, TensorFragmentRegistry,
    Modality, create_financial_tensor, create_cognitive_tensor,
    create_agent_tensor, get_global_registry
)

logger = logging.getLogger(__name__)


class BenchmarkComplexity(Enum):
    """Benchmark complexity levels"""
    MINIMAL = "minimal"  # Basic operations, small tensors
    STANDARD = "standard"  # Typical use cases
    INTENSIVE = "intensive"  # Large tensors, complex operations
    EXTREME = "extreme"  # Maximum stress testing


@dataclass
class TensorSignatureProfile:
    """Profile defining a specific tensor signature for benchmarking"""
    name: str
    shape: TensorShape
    data_generator: Callable[[TensorShape], np.ndarray]
    expected_operations: List[SymbolicOperation]
    complexity: BenchmarkComplexity
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Results from a tensor signature benchmark"""
    signature_name: str
    operation: SymbolicOperation
    architecture: KernelArchitecture
    
    # Performance metrics
    execution_times: List[float]  # in seconds
    memory_usage_mb: List[float]
    accuracy_scores: List[float]
    
    # Computed metrics
    avg_time_ms: float = 0.0
    min_time_ms: float = 0.0
    max_time_ms: float = 0.0
    std_time_ms: float = 0.0
    p50_time_ms: float = 0.0
    p95_time_ms: float = 0.0
    p99_time_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0
    
    avg_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    avg_accuracy: float = 0.0
    
    # Validation flags
    meets_latency_target: bool = False  # <5ms
    meets_accuracy_target: bool = False  # >99%
    meets_memory_target: bool = False  # reasonable usage
    
    def __post_init__(self):
        """Compute derived metrics"""
        if self.execution_times:
            times_ms = [t * 1000 for t in self.execution_times]
            self.avg_time_ms = statistics.mean(times_ms)
            self.min_time_ms = min(times_ms)
            self.max_time_ms = max(times_ms)
            self.std_time_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
            self.p50_time_ms = statistics.median(times_ms)
            
            if len(times_ms) >= 20:
                sorted_times = sorted(times_ms)
                self.p95_time_ms = sorted_times[int(len(sorted_times) * 0.95)]
                self.p99_time_ms = sorted_times[int(len(sorted_times) * 0.99)]
            
            self.throughput_ops_per_sec = 1.0 / statistics.mean(self.execution_times)
            
            # Validation checks
            self.meets_latency_target = self.avg_time_ms < 5.0
        
        if self.memory_usage_mb:
            self.avg_memory_mb = statistics.mean(self.memory_usage_mb)
            self.peak_memory_mb = max(self.memory_usage_mb)
            self.meets_memory_target = self.peak_memory_mb < 1000  # < 1GB
        
        if self.accuracy_scores:
            self.avg_accuracy = statistics.mean(self.accuracy_scores)
            self.meets_accuracy_target = self.avg_accuracy > 0.99


class TensorSignatureBenchmarkSuite:
    """Comprehensive benchmarking suite for tensor signatures"""
    
    def __init__(self):
        self.kernel_manager = get_kernel_manager()
        self.tensor_registry = get_global_registry()
        self.benchmark_results: Dict[str, List[BenchmarkResult]] = {}
        self.system_info = self._collect_system_info()
        
        # Initialize tensor signature profiles
        self.signature_profiles = self._create_signature_profiles()
        
        logger.info(f"Initialized TensorSignatureBenchmarkSuite with {len(self.signature_profiles)} profiles")
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for benchmarking context"""
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'available_memory_gb': round(psutil.virtual_memory().available / (1024**3), 2)
        }
    
    def _create_signature_profiles(self) -> List[TensorSignatureProfile]:
        """Create comprehensive set of tensor signature profiles for benchmarking"""
        profiles = []
        
        # Financial tensor signatures
        profiles.extend([
            TensorSignatureProfile(
                name="financial_small",
                shape=create_financial_tensor(depth=4, context=8, salience=6),
                data_generator=self._generate_financial_data,
                expected_operations=[SymbolicOperation.SYMBOL_ADD, SymbolicOperation.PATTERN_RECOGNITION],
                complexity=BenchmarkComplexity.MINIMAL,
                metadata={'domain': 'financial', 'use_case': 'account_analysis'}
            ),
            TensorSignatureProfile(
                name="financial_standard", 
                shape=create_financial_tensor(depth=8, context=16, salience=10),
                data_generator=self._generate_financial_data,
                expected_operations=[SymbolicOperation.SYMBOL_ADD, SymbolicOperation.SYMBOL_MULTIPLY,
                                   SymbolicOperation.PATTERN_RECOGNITION, SymbolicOperation.TENSOR_TO_SYMBOL],
                complexity=BenchmarkComplexity.STANDARD,
                metadata={'domain': 'financial', 'use_case': 'portfolio_analysis'}
            ),
            TensorSignatureProfile(
                name="financial_large",
                shape=create_financial_tensor(depth=12, context=24, salience=12),
                data_generator=self._generate_financial_data,
                expected_operations=[SymbolicOperation.SYMBOL_ADD, SymbolicOperation.SYMBOL_MULTIPLY,
                                   SymbolicOperation.PATTERN_RECOGNITION, SymbolicOperation.ATOM_EMBEDDING],
                complexity=BenchmarkComplexity.INTENSIVE,
                metadata={'domain': 'financial', 'use_case': 'market_prediction'}
            )
        ])
        
        # Cognitive tensor signatures
        profiles.extend([
            TensorSignatureProfile(
                name="cognitive_reasoning",
                shape=create_cognitive_tensor(depth=10, context=20, autonomy=6),
                data_generator=self._generate_cognitive_data,
                expected_operations=[SymbolicOperation.TENSOR_TO_SYMBOL, SymbolicOperation.SYMBOL_TO_TENSOR,
                                   SymbolicOperation.PATTERN_RECOGNITION, SymbolicOperation.ATOM_EMBEDDING],
                complexity=BenchmarkComplexity.STANDARD,
                metadata={'domain': 'cognitive', 'use_case': 'logical_reasoning'}
            ),
            TensorSignatureProfile(
                name="cognitive_complex",
                shape=create_cognitive_tensor(depth=15, context=31, autonomy=7),
                data_generator=self._generate_cognitive_data,
                expected_operations=[SymbolicOperation.SYMBOL_COMPOSE, SymbolicOperation.SYMBOL_MATCH,
                                   SymbolicOperation.PATTERN_RECOGNITION, SymbolicOperation.ATOM_EMBEDDING],
                complexity=BenchmarkComplexity.INTENSIVE,
                metadata={'domain': 'cognitive', 'use_case': 'complex_reasoning'}
            )
        ])
        
        # Agent tensor signatures
        profiles.extend([
            TensorSignatureProfile(
                name="agent_financial",
                shape=create_agent_tensor(agent_type="financial", autonomy=5),
                data_generator=self._generate_agent_data,
                expected_operations=[SymbolicOperation.SYMBOL_ADD, SymbolicOperation.TENSOR_TO_SYMBOL],
                complexity=BenchmarkComplexity.STANDARD,
                metadata={'domain': 'agent', 'use_case': 'financial_agent'}
            ),
            TensorSignatureProfile(
                name="agent_autonomous",
                shape=create_agent_tensor(agent_type="cognitive", autonomy=7),
                data_generator=self._generate_agent_data,
                expected_operations=[SymbolicOperation.SYMBOL_MULTIPLY, SymbolicOperation.PATTERN_RECOGNITION],
                complexity=BenchmarkComplexity.INTENSIVE,
                metadata={'domain': 'agent', 'use_case': 'autonomous_reasoning'}
            )
        ])
        
        # Mixed modality signatures for comprehensive testing
        profiles.extend([
            TensorSignatureProfile(
                name="mixed_temporal",
                shape=TensorShape(modality=Modality.TEMPORAL.value, depth=8, context=20, salience=10, autonomy_index=4),
                data_generator=self._generate_temporal_data,
                expected_operations=[SymbolicOperation.PATTERN_RECOGNITION, SymbolicOperation.TEMPORAL_REASONING],
                complexity=BenchmarkComplexity.STANDARD,
                metadata={'domain': 'temporal', 'use_case': 'time_series_analysis'}
            ),
            TensorSignatureProfile(
                name="mixed_linguistic",
                shape=TensorShape(modality=Modality.LINGUISTIC.value, depth=6, context=25, salience=8, autonomy_index=3),
                data_generator=self._generate_linguistic_data,
                expected_operations=[SymbolicOperation.SYMBOL_TO_TENSOR, SymbolicOperation.ATOM_EMBEDDING],
                complexity=BenchmarkComplexity.STANDARD,
                metadata={'domain': 'linguistic', 'use_case': 'text_processing'}
            )
        ])
        
        return profiles
    
    def _generate_financial_data(self, shape: TensorShape) -> np.ndarray:
        """Generate realistic financial tensor data"""
        # Use modality + 1 to avoid zero-sized tensors
        modality_dim = max(shape.modality + 1, 1)
        total_size = modality_dim * shape.depth * shape.context * shape.salience * shape.autonomy_index
        
        # Generate financial-like time series with trends and volatility
        base_values = np.random.normal(100.0, 15.0, total_size)  # Price-like values
        trend = np.linspace(-5, 5, total_size)  # Market trend
        volatility = np.random.normal(0, 0.1, total_size)  # Market volatility
        
        data = base_values + trend + volatility
        return data.reshape(modality_dim, shape.depth, shape.context, shape.salience, shape.autonomy_index).astype(np.float32)
    
    def _generate_cognitive_data(self, shape: TensorShape) -> np.ndarray:
        """Generate realistic cognitive tensor data"""
        # Use modality + 1 to avoid zero-sized tensors
        modality_dim = max(shape.modality + 1, 1)
        total_size = modality_dim * shape.depth * shape.context * shape.salience * shape.autonomy_index
        
        # Generate cognitive activation patterns
        # Higher values indicate stronger activation
        activation_strength = np.random.beta(2, 5, total_size)  # Skewed towards lower activations
        context_influence = np.random.exponential(0.3, total_size)  # Context decay
        attention_weights = np.random.gamma(2, 0.5, total_size)  # Attention patterns
        
        data = activation_strength * context_influence * attention_weights
        return data.reshape(modality_dim, shape.depth, shape.context, shape.salience, shape.autonomy_index).astype(np.float32)
    
    def _generate_agent_data(self, shape: TensorShape) -> np.ndarray:
        """Generate realistic agent state tensor data"""
        # Use modality + 1 to avoid zero-sized tensors
        modality_dim = max(shape.modality + 1, 1)
        total_size = modality_dim * shape.depth * shape.context * shape.salience * shape.autonomy_index
        
        # Generate agent state representations
        confidence_scores = np.random.uniform(0.3, 0.95, total_size)  # Agent confidence
        autonomy_levels = np.random.uniform(0.4, 1.0, total_size)  # Autonomy degrees
        decision_weights = np.random.normal(0.5, 0.2, total_size)  # Decision biases
        
        data = confidence_scores * autonomy_levels + decision_weights * 0.1
        return np.clip(data, 0.0, 1.0).reshape(modality_dim, shape.depth, shape.context, shape.salience, shape.autonomy_index).astype(np.float32)
    
    def _generate_temporal_data(self, shape: TensorShape) -> np.ndarray:
        """Generate realistic temporal tensor data"""
        # Use modality + 1 to avoid zero-sized tensors
        modality_dim = max(shape.modality + 1, 1)
        total_size = modality_dim * shape.depth * shape.context * shape.salience * shape.autonomy_index
        
        # Generate temporal sequences
        time_points = np.linspace(0, 2*np.pi, total_size)
        seasonal_pattern = np.sin(time_points) + 0.5 * np.cos(2 * time_points)
        noise = np.random.normal(0, 0.1, total_size)
        trend = np.linspace(0, 1, total_size)
        
        data = seasonal_pattern + noise + trend * 0.5
        return data.reshape(modality_dim, shape.depth, shape.context, shape.salience, shape.autonomy_index).astype(np.float32)
    
    def _generate_linguistic_data(self, shape: TensorShape) -> np.ndarray:
        """Generate realistic linguistic tensor data"""
        # Use modality + 1 to avoid zero-sized tensors
        modality_dim = max(shape.modality + 1, 1)
        total_size = modality_dim * shape.depth * shape.context * shape.salience * shape.autonomy_index
        
        # Generate linguistic feature vectors
        word_embeddings = np.random.normal(0, 0.3, total_size)  # Word embedding-like
        semantic_weights = np.random.exponential(1, total_size)  # Semantic importance
        syntactic_features = np.random.uniform(-1, 1, total_size)  # Syntactic features
        
        data = word_embeddings * semantic_weights * 0.1 + syntactic_features * 0.3
        return data.reshape(modality_dim, shape.depth, shape.context, shape.salience, shape.autonomy_index).astype(np.float32)
    
    async def benchmark_tensor_signature(self, 
                                       profile: TensorSignatureProfile,
                                       iterations: int = 50,
                                       architectures: Optional[List[KernelArchitecture]] = None) -> List[BenchmarkResult]:
        """Benchmark a specific tensor signature across operations and architectures"""
        
        if architectures is None:
            architectures = self.kernel_manager.get_available_architectures()
        
        results = []
        
        logger.info(f"Benchmarking tensor signature: {profile.name}")
        
        for architecture in architectures:
            for operation in profile.expected_operations:
                try:
                    result = await self._benchmark_operation_on_signature(
                        profile, operation, architecture, iterations
                    )
                    results.append(result)
                    
                    logger.info(f"✅ {profile.name} + {operation.name} on {architecture.value}: "
                              f"{result.avg_time_ms:.3f}ms avg, {result.avg_accuracy:.1%} accuracy")
                    
                except Exception as e:
                    logger.warning(f"❌ {profile.name} + {operation.name} on {architecture.value}: {e}")
                    continue
        
        # Store results
        if profile.name not in self.benchmark_results:
            self.benchmark_results[profile.name] = []
        self.benchmark_results[profile.name].extend(results)
        
        return results
    
    async def _benchmark_operation_on_signature(self,
                                              profile: TensorSignatureProfile,
                                              operation: SymbolicOperation,
                                              architecture: KernelArchitecture,
                                              iterations: int) -> BenchmarkResult:
        """Benchmark a specific operation on a tensor signature"""
        
        execution_times = []
        memory_usage_mb = []
        accuracy_scores = []
        
        for i in range(iterations):
            # Generate fresh test data for each iteration
            test_data = profile.data_generator(profile.shape)
            test_tensor = SymbolicTensor(
                data=test_data,
                symbols={'iteration': i, 'signature': profile.name, 'domain': profile.metadata.get('domain', 'unknown')}
            )
            
            # Measure memory before operation
            gc.collect()  # Force garbage collection
            memory_before = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            
            # Execute operation with timing
            start_time = time.perf_counter()
            
            try:
                if operation in [SymbolicOperation.SYMBOL_ADD, SymbolicOperation.SYMBOL_MULTIPLY]:
                    # Create second tensor for binary operations
                    test_data2 = profile.data_generator(profile.shape)
                    test_tensor2 = SymbolicTensor(data=test_data2, symbols={'second': True})
                    inputs = [test_tensor, test_tensor2]
                else:
                    inputs = [test_tensor]
                
                result = await self.kernel_manager.execute_operation(operation, inputs, architecture=architecture)
                
                execution_time = time.perf_counter() - start_time
                
                # Measure memory after operation
                memory_after = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
                memory_used = memory_after - memory_before
                
                # Calculate accuracy score based on operation type
                accuracy = self._calculate_accuracy_score(test_tensor, result, operation)
                
                execution_times.append(execution_time)
                memory_usage_mb.append(memory_used)
                accuracy_scores.append(accuracy)
                
            except Exception as e:
                logger.warning(f"Iteration {i} failed: {e}")
                # Record failed iteration with high time and low accuracy
                execution_times.append(0.1)  # 100ms penalty for failure
                memory_usage_mb.append(0.0)
                accuracy_scores.append(0.0)
        
        return BenchmarkResult(
            signature_name=profile.name,
            operation=operation,
            architecture=architecture,
            execution_times=execution_times,
            memory_usage_mb=memory_usage_mb,
            accuracy_scores=accuracy_scores
        )
    
    def _calculate_accuracy_score(self, input_tensor: SymbolicTensor, 
                                 result_tensor: SymbolicTensor, 
                                 operation: SymbolicOperation) -> float:
        """Calculate accuracy score for an operation result"""
        try:
            if operation == SymbolicOperation.SYMBOL_ADD:
                # For addition, check if result has expected sum properties
                if len(input_tensor.data.shape) > 0 and len(result_tensor.data.shape) > 0:
                    expected_mean = np.mean(input_tensor.data) * 2  # Assuming two similar inputs
                    actual_mean = np.mean(result_tensor.data)
                    accuracy = 1.0 - abs(expected_mean - actual_mean) / max(abs(expected_mean), 1e-6)
                    return max(0.0, min(1.0, accuracy))
            
            elif operation == SymbolicOperation.PATTERN_RECOGNITION:
                # For pattern recognition, check if patterns were detected
                if 'symmetry' in result_tensor.symbols or any('period_' in key for key in result_tensor.symbols.keys()):
                    return 1.0
                return 0.8  # Partial credit if operation succeeded but no patterns found
            
            elif operation == SymbolicOperation.TENSOR_TO_SYMBOL:
                # For tensor to symbol, check if statistical features were extracted
                expected_features = ['mean', 'std', 'range', 'energy']
                found_features = sum(1 for feature in expected_features if feature in result_tensor.symbols)
                return found_features / len(expected_features)
            
            elif operation == SymbolicOperation.ATOM_EMBEDDING:
                # For atom embedding, check if embeddings were generated
                embedding_keys = [k for k in result_tensor.symbols.keys() if '_embedding' in k]
                if embedding_keys and result_tensor.data.size > 0:
                    return 1.0
                return 0.5
            
            else:
                # For other operations, basic success check
                if result_tensor.data.size > 0 and not np.any(np.isnan(result_tensor.data)):
                    return 1.0
                return 0.0
        
        except Exception as e:
            logger.debug(f"Accuracy calculation failed: {e}")
            return 0.0
    
    async def run_comprehensive_benchmark(self, 
                                        complexity_filter: Optional[BenchmarkComplexity] = None,
                                        iterations: int = 50) -> Dict[str, Any]:
        """Run comprehensive benchmark suite across all tensor signatures"""
        
        # Filter profiles by complexity if specified
        if complexity_filter:
            profiles_to_test = [p for p in self.signature_profiles if p.complexity == complexity_filter]
        else:
            profiles_to_test = self.signature_profiles
        
        logger.info(f"Running comprehensive benchmark with {len(profiles_to_test)} profiles, {iterations} iterations each")
        
        start_time = time.time()
        total_benchmarks = 0
        successful_benchmarks = 0
        
        for profile in profiles_to_test:
            try:
                results = await self.benchmark_tensor_signature(profile, iterations)
                total_benchmarks += len(results)
                successful_benchmarks += len([r for r in results if r.meets_latency_target and r.meets_accuracy_target])
            except Exception as e:
                logger.error(f"Failed to benchmark profile {profile.name}: {e}")
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        report = {
            'benchmark_summary': {
                'total_profiles': len(profiles_to_test),
                'total_benchmarks': total_benchmarks,
                'successful_benchmarks': successful_benchmarks,
                'success_rate': successful_benchmarks / max(total_benchmarks, 1),
                'total_time_seconds': total_time,
                'iterations_per_benchmark': iterations
            },
            'system_info': self.system_info,
            'performance_targets': {
                'latency_target_ms': 5.0,
                'accuracy_target_percent': 99.0,
                'memory_target_mb': 1000
            },
            'detailed_results': self._generate_detailed_results_summary(),
            'cross_platform_analysis': self._analyze_cross_platform_performance(),
            'optimization_recommendations': self._generate_optimization_recommendations()
        }
        
        logger.info(f"Comprehensive benchmark complete: {successful_benchmarks}/{total_benchmarks} successful "
                   f"({report['benchmark_summary']['success_rate']:.1%})")
        
        return report
    
    def _generate_detailed_results_summary(self) -> Dict[str, Any]:
        """Generate detailed summary of all benchmark results"""
        summary = {
            'by_signature': {},
            'by_operation': {},
            'by_architecture': {},
            'aggregate_metrics': {
                'avg_latency_ms': 0.0,
                'min_latency_ms': float('inf'),
                'max_latency_ms': 0.0,
                'avg_accuracy': 0.0,
                'avg_memory_mb': 0.0
            }
        }
        
        all_results = []
        for signature_results in self.benchmark_results.values():
            all_results.extend(signature_results)
        
        if not all_results:
            return summary
        
        # Aggregate metrics (convert to float for statistics module compatibility)
        latencies = [float(r.avg_time_ms) for r in all_results]
        accuracies = [float(r.avg_accuracy) for r in all_results]
        memories = [float(r.avg_memory_mb) for r in all_results]
        
        summary['aggregate_metrics'] = {
            'avg_latency_ms': statistics.mean(latencies),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'avg_accuracy': statistics.mean(accuracies),
            'avg_memory_mb': statistics.mean(memories),
            'total_results': len(all_results)
        }
        
        # Group by signature
        for signature_name, results in self.benchmark_results.items():
            if results:
                signature_latencies = [float(r.avg_time_ms) for r in results]
                signature_accuracies = [float(r.avg_accuracy) for r in results]
                
                summary['by_signature'][signature_name] = {
                    'avg_latency_ms': statistics.mean(signature_latencies),
                    'avg_accuracy': statistics.mean(signature_accuracies),
                    'num_operations': len(results),
                    'meets_targets': all(r.meets_latency_target and r.meets_accuracy_target for r in results)
                }
        
        # Group by operation
        op_groups = {}
        for result in all_results:
            op_name = result.operation.name
            if op_name not in op_groups:
                op_groups[op_name] = []
            op_groups[op_name].append(result)
        
        for op_name, results in op_groups.items():
            op_latencies = [float(r.avg_time_ms) for r in results]
            op_accuracies = [float(r.avg_accuracy) for r in results]
            
            summary['by_operation'][op_name] = {
                'avg_latency_ms': statistics.mean(op_latencies),
                'avg_accuracy': statistics.mean(op_accuracies),
                'num_signatures': len(results),
                'success_rate': sum(1 for r in results if r.meets_latency_target and r.meets_accuracy_target) / len(results)
            }
        
        # Group by architecture
        arch_groups = {}
        for result in all_results:
            arch_name = result.architecture.value
            if arch_name not in arch_groups:
                arch_groups[arch_name] = []
            arch_groups[arch_name].append(result)
        
        for arch_name, results in arch_groups.items():
            arch_latencies = [float(r.avg_time_ms) for r in results]
            arch_accuracies = [float(r.avg_accuracy) for r in results]
            
            summary['by_architecture'][arch_name] = {
                'avg_latency_ms': statistics.mean(arch_latencies),
                'avg_accuracy': statistics.mean(arch_accuracies),
                'num_benchmarks': len(results),
                'efficiency_score': statistics.mean([1000.0 / max(r.avg_time_ms, 0.001) for r in results])
            }
        
        return summary
    
    def _analyze_cross_platform_performance(self) -> Dict[str, Any]:
        """Analyze performance consistency across platforms"""
        analysis = {
            'variance_analysis': {},
            'platform_ranking': {},
            'consistency_score': 0.0
        }
        
        # Group results by operation and signature, then compare across architectures
        operation_signature_groups = {}
        
        for signature_name, results in self.benchmark_results.items():
            for result in results:
                key = f"{result.operation.name}_{signature_name}"
                if key not in operation_signature_groups:
                    operation_signature_groups[key] = {}
                operation_signature_groups[key][result.architecture.value] = result
        
        # Calculate variance for each operation+signature combination
        variances = []
        for group_key, arch_results in operation_signature_groups.items():
            if len(arch_results) > 1:  # Need multiple architectures to compare
                latencies = [result.avg_time_ms for result in arch_results.values()]
                variance = statistics.variance(latencies) if len(latencies) > 1 else 0.0
                mean_latency = statistics.mean(latencies)
                relative_variance = variance / (mean_latency ** 2) if mean_latency > 0 else 0.0
                
                variances.append(relative_variance)
                analysis['variance_analysis'][group_key] = {
                    'absolute_variance': variance,
                    'relative_variance': relative_variance,
                    'latencies_by_arch': {arch: result.avg_time_ms for arch, result in arch_results.items()}
                }
        
        # Overall consistency score (lower variance = higher consistency)
        if variances:
            avg_relative_variance = statistics.mean(variances)
            # Convert to consistency score (0-1, higher is better)
            analysis['consistency_score'] = max(0.0, 1.0 - min(avg_relative_variance * 10, 1.0))
        
        return analysis
    
    def _generate_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate performance optimization recommendations based on benchmark results"""
        recommendations = []
        
        all_results = []
        for signature_results in self.benchmark_results.values():
            all_results.extend(signature_results)
        
        if not all_results:
            return recommendations
        
        # Analyze slow operations
        slow_results = [r for r in all_results if r.avg_time_ms > 5.0]
        if slow_results:
            slow_operations = {}
            for result in slow_results:
                op_name = result.operation.name
                if op_name not in slow_operations:
                    slow_operations[op_name] = []
                slow_operations[op_name].append(result)
            
            for op_name, results in slow_operations.items():
                avg_latency = statistics.mean([r.avg_time_ms for r in results])
                recommendations.append({
                    'type': 'performance_optimization',
                    'priority': 'high' if avg_latency > 10.0 else 'medium',
                    'operation': op_name,
                    'issue': f'Operation {op_name} has average latency of {avg_latency:.2f}ms (target: <5ms)',
                    'suggestions': [
                        'Consider kernel-level optimizations for this operation',
                        'Evaluate tensor size reduction strategies',
                        'Implement operation-specific caching',
                        'Consider GPU acceleration if available'
                    ]
                })
        
        # Analyze low accuracy operations
        low_accuracy_results = [r for r in all_results if r.avg_accuracy < 0.99]
        if low_accuracy_results:
            accuracy_issues = {}
            for result in low_accuracy_results:
                op_name = result.operation.name
                if op_name not in accuracy_issues:
                    accuracy_issues[op_name] = []
                accuracy_issues[op_name].append(result)
            
            for op_name, results in accuracy_issues.items():
                avg_accuracy = statistics.mean([r.avg_accuracy for r in results])
                recommendations.append({
                    'type': 'accuracy_improvement',
                    'priority': 'high' if avg_accuracy < 0.95 else 'medium',
                    'operation': op_name,
                    'issue': f'Operation {op_name} has average accuracy of {avg_accuracy:.1%} (target: >99%)',
                    'suggestions': [
                        'Review operation implementation for numerical precision issues',
                        'Validate input data quality and preprocessing',
                        'Consider algorithmic improvements for this operation type',
                        'Implement more robust error handling and validation'
                    ]
                })
        
        # Analyze memory usage
        high_memory_results = [r for r in all_results if r.peak_memory_mb > 500]
        if high_memory_results:
            recommendations.append({
                'type': 'memory_optimization',
                'priority': 'medium',
                'operations': list(set([r.operation.name for r in high_memory_results])),
                'issue': f'{len(high_memory_results)} operations have high memory usage (>500MB)',
                'suggestions': [
                    'Implement memory pooling for tensor operations',
                    'Consider streaming processing for large tensors',
                    'Optimize data structures for memory efficiency',
                    'Implement garbage collection strategies'
                ]
            })
        
        # Architecture-specific recommendations
        arch_performance = {}
        for result in all_results:
            arch = result.architecture.value
            if arch not in arch_performance:
                arch_performance[arch] = []
            arch_performance[arch].append(result.avg_time_ms)
        
        for arch, latencies in arch_performance.items():
            avg_latency = statistics.mean(latencies)
            if avg_latency > 8.0:  # Significantly above target
                recommendations.append({
                    'type': 'architecture_optimization',
                    'priority': 'medium',
                    'architecture': arch,
                    'issue': f'Architecture {arch} has average latency of {avg_latency:.2f}ms',
                    'suggestions': [
                        f'Optimize kernel implementations for {arch}',
                        'Consider architecture-specific compiler flags',
                        'Evaluate hardware-specific optimizations',
                        'Benchmark against alternative architectures'
                    ]
                })
        
        return recommendations
    
    def export_results(self, filepath: str) -> None:
        """Export benchmark results to JSON file"""
        export_data = {
            'system_info': self.system_info,
            'timestamp': time.time(),
            'results': {}
        }
        
        # Convert results to JSON-serializable format
        for signature_name, results in self.benchmark_results.items():
            export_data['results'][signature_name] = []
            for result in results:
                export_data['results'][signature_name].append({
                    'operation': result.operation.name,
                    'architecture': result.architecture.value,
                    'avg_time_ms': result.avg_time_ms,
                    'p95_time_ms': result.p95_time_ms,
                    'throughput_ops_per_sec': result.throughput_ops_per_sec,
                    'avg_accuracy': result.avg_accuracy,
                    'avg_memory_mb': result.avg_memory_mb,
                    'meets_latency_target': result.meets_latency_target,
                    'meets_accuracy_target': result.meets_accuracy_target,
                    'meets_memory_target': result.meets_memory_target
                })
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Benchmark results exported to {filepath}")


# Factory function for easy instantiation
def create_benchmark_suite() -> TensorSignatureBenchmarkSuite:
    """Create and return a configured tensor signature benchmark suite"""
    return TensorSignatureBenchmarkSuite()