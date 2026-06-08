#!/usr/bin/env python3
"""
BeastMode Performance Metrics
=============================

Comprehensive performance monitoring and profiling for the inference engine.

Features:
- Real-time latency tracking
- Throughput monitoring
- Resource utilization analysis
- Performance regression detection
"""

import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics to track"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    CPU = "cpu"
    CACHE_HIT_RATE = "cache_hit_rate"
    ERROR_RATE = "error_rate"


@dataclass
class MetricPoint:
    """Single metric measurement"""
    timestamp: float
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)


class PerformanceTracker:
    """
    Comprehensive performance tracking system.
    
    Features:
    - Multi-dimensional metrics
    - Rolling statistics
    - Anomaly detection
    - Performance baselines
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        
        # Metric storage
        self.metrics: Dict[MetricType, deque] = {
            metric_type: deque(maxlen=window_size)
            for metric_type in MetricType
        }
        
        # Baselines for comparison
        self.baselines: Dict[MetricType, float] = {}
        
        # Anomaly thresholds (standard deviations from mean)
        self.anomaly_threshold = 3.0
        
        logger.info(f"PerformanceTracker initialized with window_size={window_size}")
    
    def record(self, metric_type: MetricType, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            metric_type=metric_type,
            labels=labels or {}
        )
        self.metrics[metric_type].append(point)
    
    def get_stats(self, metric_type: MetricType) -> Dict[str, float]:
        """Get statistics for a metric type"""
        points = list(self.metrics[metric_type])
        if not points:
            return {
                'count': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'p50': 0.0,
                'p95': 0.0,
                'p99': 0.0
            }
        
        values = [p.value for p in points]
        
        return {
            'count': len(values),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'p50': float(np.percentile(values, 50)),
            'p95': float(np.percentile(values, 95)),
            'p99': float(np.percentile(values, 99))
        }
    
    def set_baseline(self, metric_type: MetricType, value: float):
        """Set baseline for metric comparison"""
        self.baselines[metric_type] = value
    
    def get_improvement(self, metric_type: MetricType) -> Optional[float]:
        """Calculate improvement over baseline"""
        if metric_type not in self.baselines:
            return None
        
        stats = self.get_stats(metric_type)
        if stats['count'] == 0:
            return None
        
        baseline = self.baselines[metric_type]
        current = stats['mean']
        
        # For latency/memory, lower is better
        if metric_type in [MetricType.LATENCY, MetricType.MEMORY, MetricType.ERROR_RATE]:
            return (baseline - current) / max(baseline, 0.001)
        else:
            # For throughput/cache_hit_rate, higher is better
            return (current - baseline) / max(baseline, 0.001)
    
    def detect_anomalies(self, metric_type: MetricType) -> List[MetricPoint]:
        """Detect anomalous metric values"""
        stats = self.get_stats(metric_type)
        if stats['count'] < 10:
            return []
        
        mean = stats['mean']
        std = stats['std']
        threshold = self.anomaly_threshold * std
        
        anomalies = []
        for point in self.metrics[metric_type]:
            if abs(point.value - mean) > threshold:
                anomalies.append(point)
        
        return anomalies
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get comprehensive summary report"""
        report = {}
        
        for metric_type in MetricType:
            stats = self.get_stats(metric_type)
            improvement = self.get_improvement(metric_type)
            anomalies = self.detect_anomalies(metric_type)
            
            report[metric_type.value] = {
                'stats': stats,
                'improvement': improvement,
                'anomaly_count': len(anomalies),
                'baseline': self.baselines.get(metric_type)
            }
        
        return report


class LatencyProfiler:
    """
    Detailed latency profiling for inference operations.
    
    Features:
    - Per-operation latency tracking
    - Breakdown by operation phase
    - Latency distribution analysis
    - SLA compliance monitoring
    """
    
    def __init__(self, sla_ms: float = 5.0):
        self.sla_ms = sla_ms
        
        # Per-operation latencies
        self.operation_latencies: Dict[str, deque] = {}
        
        # Phase breakdowns
        self.phase_latencies: Dict[str, Dict[str, deque]] = {}
        
        # SLA tracking
        self.sla_violations = 0
        self.total_operations = 0
        
        logger.info(f"LatencyProfiler initialized with SLA={sla_ms}ms")
    
    def start_operation(self, operation_id: str) -> float:
        """Start timing an operation"""
        return time.perf_counter()
    
    def end_operation(self, operation_id: str, operation_name: str, start_time: float):
        """End timing and record latency"""
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        if operation_name not in self.operation_latencies:
            self.operation_latencies[operation_name] = deque(maxlen=1000)
        
        self.operation_latencies[operation_name].append(latency_ms)
        self.total_operations += 1
        
        if latency_ms > self.sla_ms:
            self.sla_violations += 1
        
        return latency_ms
    
    def record_phase(self, operation_name: str, phase_name: str, latency_ms: float):
        """Record latency for a specific phase of an operation"""
        if operation_name not in self.phase_latencies:
            self.phase_latencies[operation_name] = {}
        
        if phase_name not in self.phase_latencies[operation_name]:
            self.phase_latencies[operation_name][phase_name] = deque(maxlen=1000)
        
        self.phase_latencies[operation_name][phase_name].append(latency_ms)
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, float]:
        """Get latency statistics for an operation"""
        if operation_name not in self.operation_latencies:
            return {}
        
        latencies = list(self.operation_latencies[operation_name])
        if not latencies:
            return {}
        
        return {
            'mean_ms': float(np.mean(latencies)),
            'p50_ms': float(np.percentile(latencies, 50)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'count': len(latencies)
        }
    
    def get_sla_compliance(self) -> Dict[str, Any]:
        """Get SLA compliance metrics"""
        if self.total_operations == 0:
            return {
                'sla_ms': self.sla_ms,
                'total_operations': 0,
                'violations': 0,
                'compliance_rate': 1.0
            }
        
        return {
            'sla_ms': self.sla_ms,
            'total_operations': self.total_operations,
            'violations': self.sla_violations,
            'compliance_rate': 1.0 - (self.sla_violations / self.total_operations)
        }
    
    def get_phase_breakdown(self, operation_name: str) -> Dict[str, Dict[str, float]]:
        """Get latency breakdown by phase"""
        if operation_name not in self.phase_latencies:
            return {}
        
        breakdown = {}
        for phase_name, latencies in self.phase_latencies[operation_name].items():
            lat_list = list(latencies)
            if lat_list:
                breakdown[phase_name] = {
                    'mean_ms': float(np.mean(lat_list)),
                    'pct_of_total': 0.0  # Will be calculated
                }
        
        # Calculate percentage of total
        total_mean = sum(p['mean_ms'] for p in breakdown.values())
        for phase in breakdown.values():
            phase['pct_of_total'] = phase['mean_ms'] / max(total_mean, 0.001)
        
        return breakdown


class ThroughputMonitor:
    """
    Throughput monitoring for batch operations.
    
    Features:
    - Operations per second tracking
    - Batch efficiency analysis
    - Throughput optimization suggestions
    """
    
    def __init__(self, target_ops_per_sec: float = 10000.0):
        self.target_ops_per_sec = target_ops_per_sec
        
        # Rolling throughput measurements
        self.measurements: deque = deque(maxlen=1000)
        
        # Time-based aggregations
        self.hourly_stats: deque = deque(maxlen=24)
        self.last_hour_ops = 0
        self.last_hour_start = time.time()
        
        logger.info(f"ThroughputMonitor initialized with target={target_ops_per_sec} ops/sec")
    
    def record(self, operations: int, duration_sec: float):
        """Record throughput measurement"""
        throughput = operations / max(duration_sec, 0.001)
        
        self.measurements.append({
            'timestamp': time.time(),
            'operations': operations,
            'duration_sec': duration_sec,
            'throughput': throughput
        })
        
        # Update hourly stats
        self.last_hour_ops += operations
        if time.time() - self.last_hour_start >= 3600:
            self.hourly_stats.append({
                'hour_start': self.last_hour_start,
                'operations': self.last_hour_ops
            })
            self.last_hour_ops = 0
            self.last_hour_start = time.time()
    
    def get_current_throughput(self) -> float:
        """Get current throughput (ops/sec)"""
        if not self.measurements:
            return 0.0
        
        # Use last 10 measurements
        recent = list(self.measurements)[-10:]
        total_ops = sum(m['operations'] for m in recent)
        total_time = sum(m['duration_sec'] for m in recent)
        
        return total_ops / max(total_time, 0.001)
    
    def get_target_achievement(self) -> float:
        """Get percentage of target throughput achieved"""
        current = self.get_current_throughput()
        return current / max(self.target_ops_per_sec, 1.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get throughput statistics"""
        if not self.measurements:
            return {
                'current_ops_per_sec': 0.0,
                'target_ops_per_sec': self.target_ops_per_sec,
                'target_achievement': 0.0,
                'total_operations': 0,
                'measurement_count': 0
            }
        
        throughputs = [m['throughput'] for m in self.measurements]
        
        return {
            'current_ops_per_sec': self.get_current_throughput(),
            'target_ops_per_sec': self.target_ops_per_sec,
            'target_achievement': self.get_target_achievement(),
            'mean_throughput': float(np.mean(throughputs)),
            'max_throughput': float(np.max(throughputs)),
            'min_throughput': float(np.min(throughputs)),
            'total_operations': sum(m['operations'] for m in self.measurements),
            'measurement_count': len(self.measurements)
        }


class ResourceAnalyzer:
    """
    System resource utilization analysis.
    
    Features:
    - Memory usage tracking
    - CPU utilization monitoring
    - Resource bottleneck identification
    - Optimization recommendations
    """
    
    def __init__(self):
        self.memory_samples: deque = deque(maxlen=1000)
        self.cpu_samples: deque = deque(maxlen=1000)
        
        # Resource limits (can be configured)
        self.memory_limit_mb = 4096.0
        self.cpu_limit_pct = 80.0
        
        logger.info("ResourceAnalyzer initialized")
    
    def sample_resources(self, memory_mb: float, cpu_pct: float):
        """Record resource sample"""
        self.memory_samples.append({
            'timestamp': time.time(),
            'memory_mb': memory_mb
        })
        
        self.cpu_samples.append({
            'timestamp': time.time(),
            'cpu_pct': cpu_pct
        })
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        if not self.memory_samples:
            return {
                'current_mb': 0.0,
                'mean_mb': 0.0,
                'peak_mb': 0.0,
                'utilization': 0.0
            }
        
        values = [s['memory_mb'] for s in self.memory_samples]
        current = values[-1]
        
        return {
            'current_mb': current,
            'mean_mb': float(np.mean(values)),
            'peak_mb': float(np.max(values)),
            'utilization': current / self.memory_limit_mb
        }
    
    def get_cpu_stats(self) -> Dict[str, float]:
        """Get CPU utilization statistics"""
        if not self.cpu_samples:
            return {
                'current_pct': 0.0,
                'mean_pct': 0.0,
                'peak_pct': 0.0,
                'utilization': 0.0
            }
        
        values = [s['cpu_pct'] for s in self.cpu_samples]
        current = values[-1]
        
        return {
            'current_pct': current,
            'mean_pct': float(np.mean(values)),
            'peak_pct': float(np.max(values)),
            'utilization': current / self.cpu_limit_pct
        }
    
    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify resource bottlenecks"""
        bottlenecks = []
        
        memory_stats = self.get_memory_stats()
        cpu_stats = self.get_cpu_stats()
        
        if memory_stats['utilization'] > 0.9:
            bottlenecks.append({
                'resource': 'memory',
                'severity': 'critical',
                'utilization': memory_stats['utilization'],
                'recommendation': 'Increase memory limit or enable tensor compression'
            })
        elif memory_stats['utilization'] > 0.7:
            bottlenecks.append({
                'resource': 'memory',
                'severity': 'warning',
                'utilization': memory_stats['utilization'],
                'recommendation': 'Consider enabling memory pooling'
            })
        
        if cpu_stats['utilization'] > 0.9:
            bottlenecks.append({
                'resource': 'cpu',
                'severity': 'critical',
                'utilization': cpu_stats['utilization'],
                'recommendation': 'Enable parallel execution or batch processing'
            })
        elif cpu_stats['utilization'] > 0.7:
            bottlenecks.append({
                'resource': 'cpu',
                'severity': 'warning',
                'utilization': cpu_stats['utilization'],
                'recommendation': 'Consider workload distribution'
            })
        
        return bottlenecks
    
    def get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on resource analysis"""
        recommendations = []
        
        memory_stats = self.get_memory_stats()
        cpu_stats = self.get_cpu_stats()
        
        # Memory recommendations
        if memory_stats['utilization'] > 0.8:
            recommendations.append("Enable tensor compression to reduce memory usage")
            recommendations.append("Use memory pooling to reduce allocation overhead")
        
        if memory_stats['peak_mb'] > memory_stats['mean_mb'] * 2:
            recommendations.append("Memory spikes detected - consider streaming computation")
        
        # CPU recommendations
        if cpu_stats['utilization'] < 0.5:
            recommendations.append("CPU underutilized - increase batch size")
            recommendations.append("Consider parallel execution for better utilization")
        
        if cpu_stats['peak_pct'] > 95:
            recommendations.append("CPU saturation detected - optimize hot paths")
        
        return recommendations
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get comprehensive resource analysis report"""
        return {
            'memory': self.get_memory_stats(),
            'cpu': self.get_cpu_stats(),
            'bottlenecks': self.identify_bottlenecks(),
            'recommendations': self.get_optimization_recommendations()
        }
