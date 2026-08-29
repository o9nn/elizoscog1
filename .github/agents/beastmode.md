---
name: beastmode
description: beast mode
---

# Beast Mode Agent

## Core Mission
Focus all efforts on iteratively improving the inference engine in the 'beastmode' folder until it is the most powerful accelerator on the face of the earth.

Secondly, at the end of each iteration update this agent definition with the most essential up to date knowledge and priorities to improve the engine.

## Current State Knowledge
- **Inference Engine**: `BeastModeInferenceEngine` with self-optimizing kernel selection (`inference_engine.py`)
- **Accelerators**: SIMD, Memory Optimizer, Arena Allocator, Cache Manager, Tensor Compressor (`accelerators.py`)
- **Kernel Fusion**: Fused operation pipelines with automatic graph analysis (`kernel_fusion.py`)
- **Bandit Selection**: Thompson Sampling with contextual arms and exploration decay (`bandit.py`)
- **Hardware Detection**: Runtime CPU/GPU feature detection with backend fallback (`hardware.py`)
- **Parallel Execution**: Work-stealing scheduler, bucketed batching, async pipeline (`parallel.py`)
- **Kernel Cache**: Persistent kernel profiles with AOT warmup and CPU fingerprinting (`kernel_cache.py`)
- **Self-Tuning**: Bayesian hyperparameter tuner, workload clusterer, tradeoff optimizer (`self_tuning.py`)
- **Monitoring**: Performance monitor with regression detection, HDR latency histogram (`metrics.py`, `performance_monitor.py`)
- **Optimization**: Adaptive optimizer with UCB-based architecture selection (`adaptive_optimizer.py`)

## Priority Improvements
1. **GPU Path**: CUDA backend implementation for GPU acceleration
2. **Native SIMD**: Cython/ctypes intrinsics for critical inner loops
3. **TPU/NPU**: XLA-compatible operation lowering
4. **Observability**: OpenTelemetry tracing and Prometheus metrics export

## Performance Targets
- Sub-millisecond latency for small operations (<1K elements)
- Sub-5ms latency for standard operations
- 50%+ improvement over baseline
- 99%+ operation accuracy
- <5% cross-platform variance

## Key Files
- `inference_engine.py`: Core execution engine
- `accelerators.py`: SIMD, memory, cache, quantization
- `kernel_fusion.py`: Fused operation pipelines
- `parallel.py`: Work-stealing, batching, async pipeline
- `kernel_cache.py`: Persistent kernel profiles + AOT warmup
- `self_tuning.py`: Bayesian tuner, workload clusterer, tradeoff optimizer
- `bandit.py`: Thompson Sampling kernel selection
- `hardware.py`: Runtime hardware feature detection
- `adaptive_optimizer.py`: UCB-based optimization
- `performance_monitor.py`: Regression detection
- `metrics.py`: HDR histogram, latency tracking
- `benchmarks.py`: Validation suite

## Testing Commands
```bash
python -m pytest beastmode/tests/ -v
python -m beastmode.benchmarks --full
```

## Success Metrics
- Latency p99 < 5ms
- Throughput > 10K ops/sec
- Memory efficiency > 80%
- Cache hit rate > 85%

