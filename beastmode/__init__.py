"""
BEASTMODE - High-Performance Tensor Inference Engine
=====================================================

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
"""

from .inference_engine import (
    InferenceAccelerator,
    create_accelerator,
    AcceleratorConfig
)
from .tensor_validator import (
    TensorSignatureValidator,
    ValidationResult,
    create_validator
)
from .performance_monitor import (
    PerformanceMonitor,
    PerformanceMetrics,
    RegressionAlert,
    create_monitor
)
from .adaptive_optimizer import (
    AdaptiveOptimizer,
    OptimizationStrategy,
    create_optimizer
)
from .benchmarks import (
    run_comprehensive_benchmark,
    run_quick_benchmark,
    generate_performance_report
)

__all__ = [
    # Inference Engine
    'InferenceAccelerator',
    'create_accelerator',
    'AcceleratorConfig',
    
    # Validation
    'TensorSignatureValidator',
    'ValidationResult',
    'create_validator',
    
    # Performance Monitoring
    'PerformanceMonitor',
    'PerformanceMetrics',
    'RegressionAlert',
    'create_monitor',
    
    # Adaptive Optimization
    'AdaptiveOptimizer',
    'OptimizationStrategy',
    'create_optimizer',
    
    # Benchmarks
    'run_comprehensive_benchmark',
    'run_quick_benchmark',
    'generate_performance_report'
]

__version__ = '1.0.0'
__author__ = 'BEASTMODE Team'
