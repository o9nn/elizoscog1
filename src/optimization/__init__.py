"""
Phase 4: Optimization and Scaling - Performance Optimization Module
"""

from .performance_optimization import (
    PerformanceProfiler,
    CachingStrategy,
    DistributedProcessingEngine
)

from .production_readiness import (
    MonitoringSystem,
    BackupManager,
    DeploymentAutomation
)

from .adaptive_optimization import (
    AdaptiveParameter,
    AdaptiveStrategy,
    FitnessLandscapeType,
    ContinuousBenchmarkConfig,
    ContinuousPerformanceBenchmark,
    SelfTuningAlgorithm,
    FitnessLandscapeMapper,
    AdaptiveOptimizationEngine,
    create_adaptive_optimization_engine
)

from .optimization_reporting import (
    OptimizationTrajectoryVisualizer,
    EvolutionaryOptimizationReporter
)

__all__ = [
    'PerformanceProfiler',
    'CachingStrategy', 
    'DistributedProcessingEngine',
    'MonitoringSystem',
    'BackupManager',
    'DeploymentAutomation',
    'AdaptiveParameter',
    'AdaptiveStrategy',
    'FitnessLandscapeType',
    'ContinuousBenchmarkConfig',
    'ContinuousPerformanceBenchmark',
    'SelfTuningAlgorithm',
    'FitnessLandscapeMapper',
    'AdaptiveOptimizationEngine',
    'create_adaptive_optimization_engine',
    'OptimizationTrajectoryVisualizer',
    'EvolutionaryOptimizationReporter'
]