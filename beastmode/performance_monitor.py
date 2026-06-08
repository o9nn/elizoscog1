#!/usr/bin/env python3
"""
BEASTMODE Performance Monitor
==============================

Continuous performance monitoring with regression detection and alerting.
"""

import asyncio
import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.ggml_symbolic_kernels import (
    SymbolicTensor, SymbolicOperation, KernelArchitecture, get_kernel_manager
)

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Severity level for performance alerts"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class MetricType(Enum):
    """Type of performance metric"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    ACCURACY = "accuracy"


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: float
    operation: str
    architecture: str
    
    latency_ms: float
    throughput_ops_sec: float
    memory_mb: float
    accuracy: float
    
    cache_hit: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'operation': self.operation,
            'architecture': self.architecture,
            'latency_ms': self.latency_ms,
            'throughput_ops_sec': self.throughput_ops_sec,
            'memory_mb': self.memory_mb,
            'accuracy': self.accuracy,
            'cache_hit': self.cache_hit
        }


@dataclass
class RegressionAlert:
    """Alert for performance regression"""
    timestamp: float
    metric_type: MetricType
    operation: str
    architecture: str
    severity: AlertSeverity
    
    baseline_value: float
    current_value: float
    regression_factor: float
    
    message: str
    recommended_action: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'metric_type': self.metric_type.value,
            'operation': self.operation,
            'architecture': self.architecture,
            'severity': self.severity.value,
            'baseline_value': self.baseline_value,
            'current_value': self.current_value,
            'regression_factor': self.regression_factor,
            'message': self.message,
            'recommended_action': self.recommended_action
        }


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression detection"""
    operation: str
    architecture: str
    
    latency_mean: float
    latency_std: float
    latency_p95: float
    
    throughput_mean: float
    memory_mean: float
    accuracy_mean: float
    
    measurement_count: int
    last_updated: float
    
    # Regression thresholds
    latency_threshold: float = 1.5  # 50% increase triggers alert
    throughput_threshold: float = 0.7  # 30% decrease triggers alert
    memory_threshold: float = 2.0  # 100% increase triggers alert
    accuracy_threshold: float = 0.95  # Drop below 95% of baseline triggers alert


class PerformanceMonitor:
    """
    Real-time performance monitoring with regression detection.
    
    Features:
    - Continuous metric collection
    - Baseline management
    - Regression detection with configurable thresholds
    - Alert generation and management
    - Performance trend analysis
    """
    
    def __init__(self, 
                 history_size: int = 1000,
                 baseline_window: int = 100,
                 regression_check_interval: int = 10):
        self.kernel_manager = get_kernel_manager()
        
        # Metric storage
        self.metrics_history: Dict[str, deque] = {}
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.alerts: List[RegressionAlert] = []
        
        # Configuration
        self.history_size = history_size
        self.baseline_window = baseline_window
        self.regression_check_interval = regression_check_interval
        
        # Counters
        self.total_metrics = 0
        self.metrics_since_baseline = 0
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[RegressionAlert], None]] = []
        
        logger.info("PerformanceMonitor initialized")
    
    def _get_key(self, operation: str, architecture: str) -> str:
        """Generate key for operation+architecture combination"""
        return f"{operation}_{architecture}"
    
    async def record_metrics(self, 
                           operation: SymbolicOperation,
                           architecture: KernelArchitecture,
                           latency_ms: float,
                           memory_mb: float,
                           accuracy: float,
                           cache_hit: bool = False) -> Optional[RegressionAlert]:
        """
        Record performance metrics and check for regressions.
        
        Returns RegressionAlert if regression detected, None otherwise.
        """
        key = self._get_key(operation.name, architecture.value)
        
        # Create metrics snapshot
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            operation=operation.name,
            architecture=architecture.value,
            latency_ms=latency_ms,
            throughput_ops_sec=1000.0 / max(latency_ms, 0.001),
            memory_mb=memory_mb,
            accuracy=accuracy,
            cache_hit=cache_hit
        )
        
        # Store in history
        if key not in self.metrics_history:
            self.metrics_history[key] = deque(maxlen=self.history_size)
        self.metrics_history[key].append(metrics)
        
        self.total_metrics += 1
        self.metrics_since_baseline += 1
        
        # Update baseline periodically
        if self.metrics_since_baseline >= self.baseline_window:
            await self._update_baselines()
            self.metrics_since_baseline = 0
        
        # Check for regression
        if self.total_metrics % self.regression_check_interval == 0:
            alert = self._check_regression(key, metrics)
            if alert:
                self.alerts.append(alert)
                self._notify_alert(alert)
                return alert
        
        return None
    
    async def _update_baselines(self) -> None:
        """Update performance baselines from recent metrics"""
        for key, history in self.metrics_history.items():
            if len(history) < 10:
                continue
            
            recent = list(history)[-self.baseline_window:]
            
            latencies = [m.latency_ms for m in recent if not m.cache_hit]
            throughputs = [m.throughput_ops_sec for m in recent if not m.cache_hit]
            memories = [m.memory_mb for m in recent]
            accuracies = [m.accuracy for m in recent]
            
            if not latencies:
                continue
            
            parts = key.split('_', 1)
            operation = parts[0]
            architecture = parts[1] if len(parts) > 1 else 'unknown'
            
            self.baselines[key] = PerformanceBaseline(
                operation=operation,
                architecture=architecture,
                latency_mean=float(np.mean(latencies)),
                latency_std=float(np.std(latencies)),
                latency_p95=float(np.percentile(latencies, 95)),
                throughput_mean=float(np.mean(throughputs)),
                memory_mean=float(np.mean(memories)),
                accuracy_mean=float(np.mean(accuracies)),
                measurement_count=len(recent),
                last_updated=time.time()
            )
        
        logger.debug(f"Updated {len(self.baselines)} baselines")
    
    def _check_regression(self, key: str, metrics: PerformanceMetrics) -> Optional[RegressionAlert]:
        """Check for performance regression"""
        if key not in self.baselines:
            return None
        
        baseline = self.baselines[key]
        
        # Skip cache hits for latency checks
        if not metrics.cache_hit:
            # Check latency regression
            if metrics.latency_ms > baseline.latency_mean * baseline.latency_threshold:
                regression_factor = metrics.latency_ms / baseline.latency_mean
                severity = AlertSeverity.WARNING if regression_factor < 2.0 else AlertSeverity.CRITICAL
                
                return RegressionAlert(
                    timestamp=time.time(),
                    metric_type=MetricType.LATENCY,
                    operation=metrics.operation,
                    architecture=metrics.architecture,
                    severity=severity,
                    baseline_value=baseline.latency_mean,
                    current_value=metrics.latency_ms,
                    regression_factor=regression_factor,
                    message=f"Latency regression: {metrics.latency_ms:.2f}ms vs baseline {baseline.latency_mean:.2f}ms ({regression_factor:.1f}x)",
                    recommended_action="Check for resource contention or inefficient operations"
                )
            
            # Check throughput regression
            if metrics.throughput_ops_sec < baseline.throughput_mean * baseline.throughput_threshold:
                regression_factor = baseline.throughput_mean / max(metrics.throughput_ops_sec, 0.001)
                severity = AlertSeverity.WARNING if regression_factor < 2.0 else AlertSeverity.CRITICAL
                
                return RegressionAlert(
                    timestamp=time.time(),
                    metric_type=MetricType.THROUGHPUT,
                    operation=metrics.operation,
                    architecture=metrics.architecture,
                    severity=severity,
                    baseline_value=baseline.throughput_mean,
                    current_value=metrics.throughput_ops_sec,
                    regression_factor=regression_factor,
                    message=f"Throughput regression: {metrics.throughput_ops_sec:.1f} ops/s vs baseline {baseline.throughput_mean:.1f} ops/s",
                    recommended_action="Investigate bottlenecks and optimize hot paths"
                )
        
        # Check accuracy regression
        if metrics.accuracy < baseline.accuracy_mean * baseline.accuracy_threshold:
            regression_factor = baseline.accuracy_mean / max(metrics.accuracy, 0.001)
            severity = AlertSeverity.CRITICAL
            
            return RegressionAlert(
                timestamp=time.time(),
                metric_type=MetricType.ACCURACY,
                operation=metrics.operation,
                architecture=metrics.architecture,
                severity=severity,
                baseline_value=baseline.accuracy_mean,
                current_value=metrics.accuracy,
                regression_factor=regression_factor,
                message=f"Accuracy regression: {metrics.accuracy:.2%} vs baseline {baseline.accuracy_mean:.2%}",
                recommended_action="Review numerical precision and operation correctness"
            )
        
        return None
    
    def _notify_alert(self, alert: RegressionAlert) -> None:
        """Notify registered callbacks of alert"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.warning(f"Alert callback failed: {e}")
        
        logger.warning(f"Performance alert: [{alert.severity.value}] {alert.message}")
    
    def register_alert_callback(self, callback: Callable[[RegressionAlert], None]) -> None:
        """Register callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self, 
                           operation: Optional[str] = None,
                           architecture: Optional[str] = None) -> Dict[str, Any]:
        """Get current performance metrics"""
        results = {}
        
        for key, history in self.metrics_history.items():
            if operation and not key.startswith(operation):
                continue
            if architecture and architecture not in key:
                continue
            
            if not history:
                continue
            
            recent = list(history)[-100:]  # Last 100 metrics
            
            latencies = [m.latency_ms for m in recent if not m.cache_hit]
            throughputs = [m.throughput_ops_sec for m in recent if not m.cache_hit]
            accuracies = [m.accuracy for m in recent]
            
            if latencies:
                results[key] = {
                    'latency_mean': float(np.mean(latencies)),
                    'latency_p50': float(np.percentile(latencies, 50)),
                    'latency_p95': float(np.percentile(latencies, 95)),
                    'latency_p99': float(np.percentile(latencies, 99)) if len(latencies) >= 100 else None,
                    'throughput_mean': float(np.mean(throughputs)),
                    'accuracy_mean': float(np.mean(accuracies)),
                    'measurements': len(recent),
                    'cache_hit_rate': sum(1 for m in recent if m.cache_hit) / len(recent)
                }
        
        return results
    
    def get_performance_trends(self, window_minutes: float = 5.0) -> Dict[str, Any]:
        """Get performance trends over time window"""
        cutoff_time = time.time() - (window_minutes * 60)
        trends = {}
        
        for key, history in self.metrics_history.items():
            recent = [m for m in history if m.timestamp > cutoff_time]
            if len(recent) < 2:
                continue
            
            # Sort by timestamp
            recent.sort(key=lambda m: m.timestamp)
            
            # Calculate trend (slope of metrics over time)
            times = np.array([m.timestamp for m in recent])
            latencies = np.array([m.latency_ms for m in recent if not m.cache_hit])
            
            if len(latencies) >= 2:
                # Simple linear regression for trend
                times_norm = times[:len(latencies)] - times[0]
                slope, _ = np.polyfit(times_norm, latencies, 1)
                
                trend_direction = "stable"
                if slope > 0.1:
                    trend_direction = "degrading"
                elif slope < -0.1:
                    trend_direction = "improving"
                
                trends[key] = {
                    'direction': trend_direction,
                    'slope': float(slope),
                    'start_latency': float(latencies[0]),
                    'end_latency': float(latencies[-1]),
                    'measurement_count': len(latencies)
                }
        
        return trends
    
    def get_alerts(self, 
                  severity: Optional[AlertSeverity] = None,
                  since_timestamp: Optional[float] = None) -> List[RegressionAlert]:
        """Get performance alerts"""
        alerts = self.alerts
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if since_timestamp:
            alerts = [a for a in alerts if a.timestamp > since_timestamp]
        
        return alerts
    
    def clear_alerts(self) -> None:
        """Clear all alerts"""
        self.alerts.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary"""
        return {
            'total_metrics_recorded': self.total_metrics,
            'operations_monitored': len(self.metrics_history),
            'baselines_established': len(self.baselines),
            'active_alerts': len(self.alerts),
            'critical_alerts': len([a for a in self.alerts if a.severity == AlertSeverity.CRITICAL]),
            'warning_alerts': len([a for a in self.alerts if a.severity == AlertSeverity.WARNING]),
            'current_metrics': self.get_current_metrics()
        }
    
    def export_data(self, filepath: str) -> None:
        """Export monitoring data to file"""
        import json
        
        export_data = {
            'timestamp': time.time(),
            'summary': self.get_summary(),
            'baselines': {k: {
                'operation': v.operation,
                'architecture': v.architecture,
                'latency_mean': v.latency_mean,
                'latency_std': v.latency_std,
                'throughput_mean': v.throughput_mean,
                'accuracy_mean': v.accuracy_mean,
                'measurement_count': v.measurement_count
            } for k, v in self.baselines.items()},
            'alerts': [a.to_dict() for a in self.alerts[-100:]],  # Last 100 alerts
            'trends': self.get_performance_trends()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Performance data exported to {filepath}")


def create_monitor(history_size: int = 1000,
                  baseline_window: int = 100) -> PerformanceMonitor:
    """Factory function to create a performance monitor"""
    return PerformanceMonitor(history_size=history_size, baseline_window=baseline_window)
