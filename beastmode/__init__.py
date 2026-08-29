"""
BEASTMODE - High-Performance Tensor Inference Engine

The most powerful accelerator for cognitive computing tensor operations.

This module provides:
- High-performance tensor signature benchmarking
- Real-data validation protocols
- Cross-platform performance optimization
- Automated regression testing
- Adaptive kernel selection
- Self-tuning performance parameters

Key Components:
- TensorSignatureValidator: Real-data validation with no mocks
- InferenceAccelerator: Optimized tensor operation execution
- PerformanceMonitor: Continuous performance tracking and regression detection
- AdaptiveOptimizer: Self-tuning performance optimization
- CrossPlatformValidator: Cross-platform consistency validation
- RegressionTester: Automated regression testing
"""

from .inference_engine import (
    InferenceAccelerator,
    create_accelerator,
    AcceleratorConfig,
    ExecutionMode,
    ExecutionResult
)
from .tensor_validator import (
    TensorSignatureValidator,
    ValidationResult,
    ValidationLevel,
    create_validator
)
from .performance_monitor import (
    PerformanceMonitor,
    PerformanceMetrics,
    RegressionAlert,
    AlertSeverity,
    create_monitor
)
from .adaptive_optimizer import (
    AdaptiveOptimizer,
    OptimizationStrategy,
    OptimizationConfig,
    OptimizationResult,
    create_optimizer
)
from .cross_platform import (
    CrossPlatformValidator,
    CrossPlatformResult,
    PlatformInfo,
    create_cross_platform_validator
)
from .regression_testing import (
    RegressionTester,
    RegressionResult,
    RegressionSeverity,
    RegressionTestConfig,
    PerformanceBaseline,
    create_regression_tester
)
from .benchmarks import (
    run_comprehensive_benchmark,
    run_quick_benchmark,
    generate_performance_report,
    BenchmarkConfig
)
from .inference_engine import (
    BeastModeInferenceEngine,
    KernelSelector,
    BatchProcessor,
    ParallelExecutor,
    get_beastmode_engine
)
from .accelerators import (
    SIMDAccelerator,
    MemoryOptimizer,
    ArenaAllocator,
    CacheManager,
    TensorCompressor,
    aligned_empty,
    is_aligned
)
from .metrics import (
    PerformanceTracker,
    LatencyProfiler,
    LatencyHistogram,
    ThroughputMonitor,
    ResourceAnalyzer
)
from .hardware import (
    CPUFeatures,
    detect_cpu_features,
    detect_gpu_capabilities,
    recommend_backend
)
from .bandit import (
    ThompsonSamplingSelector,
    LatencyRewardModel,
    DecaySchedule,
    create_thompson_selector
)
from .kernel_fusion import (
    FusionPipeline,
    OperationGraph,
    create_fusion_pipeline
)
from .parallel import (
    Task,
    DependencyGraph,
    WorkStealingScheduler,
    BucketedBatcher,
    BatchSLA,
    AsyncPipeline,
    PipelineStage,
    create_work_stealing_scheduler,
    create_bucketed_batcher,
    create_async_pipeline
)
from .kernel_cache import (
    KernelProfileCache,
    SpecializedKernelProfile,
    classify_shape,
    create_kernel_cache
)
from .self_tuning import (
    BayesianTuner,
    WorkloadClusterer,
    TradeoffOptimizer,
    create_bayesian_tuner,
    create_workload_clusterer,
    create_tradeoff_optimizer
)

__all__ = [
    # Inference Engine
    'InferenceAccelerator',
    'create_accelerator',
    'AcceleratorConfig',
    'ExecutionMode',
    'ExecutionResult',
    'BeastModeInferenceEngine',
    'KernelSelector',
    'BatchProcessor',
    'ParallelExecutor',
    'get_beastmode_engine',
    
    # Validation
    'TensorSignatureValidator',
    'ValidationResult',
    'ValidationLevel',
    'create_validator',
    
    # Performance Monitoring
    'PerformanceMonitor',
    'PerformanceMetrics',
    'RegressionAlert',
    'AlertSeverity',
    'create_monitor',
    
    # Adaptive Optimization
    'AdaptiveOptimizer',
    'OptimizationStrategy',
    'OptimizationConfig',
    'OptimizationResult',
    'create_optimizer',
    
    # Cross-Platform Validation
    'CrossPlatformValidator',
    'CrossPlatformResult',
    'PlatformInfo',
    'create_cross_platform_validator',
    
    # Regression Testing
    'RegressionTester',
    'RegressionResult',
    'RegressionSeverity',
    'RegressionTestConfig',
    'PerformanceBaseline',
    'create_regression_tester',
    
    # Benchmarks
    'run_comprehensive_benchmark',
    'run_quick_benchmark',
    'generate_performance_report',
    'BenchmarkConfig',
    
    # Accelerators
    'SIMDAccelerator',
    'MemoryOptimizer',
    'ArenaAllocator',
    'CacheManager',
    'TensorCompressor',
    'aligned_empty',
    'is_aligned',
    
    # Metrics
    'PerformanceTracker',
    'LatencyProfiler',
    'LatencyHistogram',
    'ThroughputMonitor',
    'ResourceAnalyzer',
    
    # Hardware Detection
    'CPUFeatures',
    'detect_cpu_features',
    'detect_gpu_capabilities',
    'recommend_backend',
    
    # Bandit Selection
    'ThompsonSamplingSelector',
    'LatencyRewardModel',
    'DecaySchedule',
    'create_thompson_selector',
    
    # Kernel Fusion
    'FusionPipeline',
    'OperationGraph',
    'create_fusion_pipeline',

    # Parallel Execution (Phase 3)
    'Task',
    'DependencyGraph',
    'WorkStealingScheduler',
    'BucketedBatcher',
    'BatchSLA',
    'AsyncPipeline',
    'PipelineStage',
    'create_work_stealing_scheduler',
    'create_bucketed_batcher',
    'create_async_pipeline',

    # Kernel Cache (Phase 2.2)
    'KernelProfileCache',
    'SpecializedKernelProfile',
    'classify_shape',
    'create_kernel_cache',

    # Self-Tuning (Phase 5.2)
    'BayesianTuner',
    'WorkloadClusterer',
    'TradeoffOptimizer',
    'create_bayesian_tuner',
    'create_workload_clusterer',
    'create_tradeoff_optimizer'
]

__version__ = '1.2.0'
__author__ = 'BEASTMODE Team'
