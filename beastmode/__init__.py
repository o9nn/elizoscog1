#!/usr/bin/env python3
"""
BeastMode Inference Engine
============================

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
