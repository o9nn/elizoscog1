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

__all__ = [
    # Inference Engine
    'InferenceAccelerator',
    'create_accelerator',
    'AcceleratorConfig',
    'ExecutionMode',
    'ExecutionResult',
    
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
    'BenchmarkConfig'
]

__version__ = '1.0.0'
__author__ = 'BEASTMODE Team'
#!/usr/bin/env python3
"""
BeastMode Inference Engine

Ultra-high-performance neural-symbolic inference engine with:
- Self-optimizing kernel selection algorithms
- Hardware-accelerated cognitive operations
- Adaptive batch processing
- Memory-efficient symbolic computation
- Parallel kernel execution pipeline

This module provides the most powerful accelerator for cognitive computation.
"""

from .inference_engine import (
    BeastModeInferenceEngine,
    KernelSelector,
    AdaptiveOptimizer,
    BatchProcessor,
    ParallelExecutor,
    get_beastmode_engine
)

from .accelerators import (
    SIMDAccelerator,
    MemoryOptimizer,
    CacheManager,
    TensorCompressor
)

from .metrics import (
    PerformanceTracker,
    LatencyProfiler,
    ThroughputMonitor,
    ResourceAnalyzer
)

__all__ = [
    # Core Engine
    'BeastModeInferenceEngine',
    'KernelSelector',
    'AdaptiveOptimizer',
    'BatchProcessor',
    'ParallelExecutor',
    'get_beastmode_engine',
    
    # Accelerators
    'SIMDAccelerator',
    'MemoryOptimizer',
    'CacheManager',
    'TensorCompressor',
    
    # Metrics
    'PerformanceTracker',
    'LatencyProfiler',
    'ThroughputMonitor',
    'ResourceAnalyzer'
]

__version__ = '1.0.0'
__author__ = 'BeastMode Team'
