#!/usr/bin/env python3
"""
BEASTMODE Self-Tuning System
=============================

Bayesian-style hyperparameter optimization and workload characterization
for continuous self-improvement of the inference engine.

Features:
- Bayesian optimization (surrogate model + acquisition function)
- Workload characterization with online clustering
- Memory/compute tradeoff optimization
- Continuous A/B testing for optimization strategies
"""

import logging
import time
import math
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bayesian-Style Hyperparameter Tuner
# ---------------------------------------------------------------------------

@dataclass
class HyperparameterObservation:
    """One observed hyperparameter -> objective mapping"""
    params: Dict[str, float]
    objective: float  # Higher is better
    timestamp: float = field(default_factory=time.time)


class BayesianTuner:
    """
    Lightweight Bayesian-style hyperparameter optimizer.

    Uses a Gaussian-kernel surrogate model over observed
    (params, objective) pairs and an Upper Confidence Bound (UCB)
    acquisition function to balance exploration and exploitation.

    This avoids external dependencies (no scikit-learn/scipy needed)
    while still providing principled search.

    Usage:
        tuner = BayesianTuner(bounds={'learning_rate': (0.001, 1.0),
                                      'batch_size': (1, 512)})
        params = tuner.suggest()
        tuner.observe(params, objective=0.95)
    """

    def __init__(self, bounds: Dict[str, Tuple[float, float]],
                 exploration_weight: float = 0.1,
                 rng: Optional[np.random.Generator] = None):
        self.bounds = bounds
        self.param_names = sorted(bounds.keys())
        self.exploration_weight = exploration_weight
        self.rng = rng or np.random.default_rng()

        self.observations: List[HyperparameterObservation] = []
        self.best_params: Optional[Dict[str, float]] = None
        self.best_objective: float = float('-inf')

        logger.info(
            f"BayesianTuner initialized: params={self.param_names}"
        )

    def _params_to_vector(self, params: Dict[str, float]) -> np.ndarray:
        """Convert params dict to normalized vector in [0, 1]^d"""
        vec = np.zeros(len(self.param_names))
        for i, name in enumerate(self.param_names):
            lo, hi = self.bounds[name]
            val = params.get(name, (lo + hi) / 2)
            vec[i] = (val - lo) / max(hi - lo, 1e-12)
        return vec

    def _vector_to_params(self, vec: np.ndarray) -> Dict[str, float]:
        """Convert normalized vector back to params dict"""
        params = {}
        for i, name in enumerate(self.param_names):
            lo, hi = self.bounds[name]
            params[name] = lo + np.clip(vec[i], 0, 1) * (hi - lo)
        return params

    def _surrogate_predict(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Predict (mean, std) of objective at point x using
        Gaussian-kernel weighted average of observations.
        """
        if not self.observations:
            return 0.5, 1.0  # Prior

        xs = np.array([self._params_to_vector(o.params)
                       for o in self.observations])
        ys = np.array([o.objective for o in self.observations])

        # Gaussian kernel weights
        length_scale = 0.3  # Characteristic distance
        dists = np.linalg.norm(xs - x, axis=1)
        weights = np.exp(-0.5 * (dists / length_scale) ** 2)
        weight_sum = np.sum(weights)

        if weight_sum < 1e-10:
            return 0.5, 1.0

        mean = np.sum(weights * ys) / weight_sum
        variance = np.sum(weights * (ys - mean) ** 2) / weight_sum
        # Add exploration bonus for distance from observed points
        min_dist = np.min(dists)
        variance += self.exploration_weight * min_dist

        return float(mean), float(np.sqrt(max(variance, 1e-10)))

    def suggest(self, n_candidates: int = 100) -> Dict[str, float]:
        """
        Suggest next hyperparameters to try using UCB acquisition.

        Samples random candidates and picks the one with the highest
        mean + beta * std (UCB).
        """
        if not self.observations:
            # First call: return center of bounds
            return {name: (lo + hi) / 2
                    for name, (lo, hi) in self.bounds.items()}

        best_x = None
        best_acq = float('-inf')
        beta = 2.0  # UCB exploration parameter

        for _ in range(n_candidates):
            x = self.rng.random(len(self.param_names))
            mean, std = self._surrogate_predict(x)
            acquisition = mean + beta * std

            if acquisition > best_acq:
                best_acq = acquisition
                best_x = x

        return self._vector_to_params(best_x)

    def observe(self, params: Dict[str, float], objective: float) -> None:
        """Record an observation"""
        obs = HyperparameterObservation(params=params, objective=objective)
        self.observations.append(obs)

        if objective > self.best_objective:
            self.best_objective = objective
            self.best_params = dict(params)

    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_observations': len(self.observations),
            'best_objective': self.best_objective,
            'best_params': self.best_params,
            'param_names': self.param_names,
        }


# ---------------------------------------------------------------------------
# Workload Characterization (Online Clustering)
# ---------------------------------------------------------------------------

@dataclass
class WorkloadCluster:
    """Centroid and stats for one workload cluster"""
    centroid: np.ndarray
    count: int = 0
    total_latency_ms: float = 0.0
    label: str = ""

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.count, 1)


class WorkloadClusterer:
    """
    Online clustering of workload feature vectors.

    Uses a simple nearest-centroid algorithm with adaptive cluster
    creation (like leader clustering). Each incoming workload
    observation is assigned to the nearest cluster or creates a new
    one if too far from all existing centroids.

    Feature vector: (log_elements, log_latency, ndim, density)
    """

    def __init__(self, distance_threshold: float = 1.5,
                 max_clusters: int = 32):
        self.distance_threshold = distance_threshold
        self.max_clusters = max_clusters
        self.clusters: List[WorkloadCluster] = []
        self.total_observations = 0

        logger.info(
            f"WorkloadClusterer initialized: threshold={distance_threshold}"
        )

    @staticmethod
    def extract_features(shape: Tuple[int, ...],
                         latency_ms: float = 0.0,
                         density: float = 1.0) -> np.ndarray:
        """Extract a feature vector from workload characteristics"""
        total_elements = int(np.prod(shape)) if shape else 1
        return np.array([
            math.log10(max(total_elements, 1)),
            math.log10(max(latency_ms, 0.001)),
            float(len(shape)),
            density,
        ])

    def assign(self, features: np.ndarray,
               latency_ms: float = 0.0) -> int:
        """
        Assign a feature vector to the nearest cluster.

        Returns cluster index. Creates a new cluster if the nearest
        one is beyond distance_threshold.
        """
        self.total_observations += 1

        if not self.clusters:
            self.clusters.append(WorkloadCluster(
                centroid=features.copy(), count=1,
                total_latency_ms=latency_ms,
                label=f"cluster_0",
            ))
            return 0

        # Find nearest centroid
        dists = [np.linalg.norm(features - c.centroid)
                 for c in self.clusters]
        nearest_idx = int(np.argmin(dists))
        nearest_dist = dists[nearest_idx]

        if nearest_dist > self.distance_threshold and \
                len(self.clusters) < self.max_clusters:
            # Create new cluster
            idx = len(self.clusters)
            self.clusters.append(WorkloadCluster(
                centroid=features.copy(), count=1,
                total_latency_ms=latency_ms,
                label=f"cluster_{idx}",
            ))
            return idx

        # Update nearest cluster (incremental centroid update)
        cluster = self.clusters[nearest_idx]
        cluster.count += 1
        cluster.total_latency_ms += latency_ms
        # Incremental mean update
        cluster.centroid += (features - cluster.centroid) / cluster.count

        return nearest_idx

    def characterize(self, shape: Tuple[int, ...],
                     latency_ms: float = 0.0,
                     density: float = 1.0) -> Dict[str, Any]:
        """
        Characterize a workload and return cluster info.
        """
        features = self.extract_features(shape, latency_ms, density)
        cluster_idx = self.assign(features, latency_ms)
        cluster = self.clusters[cluster_idx]

        return {
            'cluster_id': cluster_idx,
            'cluster_label': cluster.label,
            'cluster_size': cluster.count,
            'cluster_avg_latency_ms': cluster.avg_latency_ms,
            'features': features.tolist(),
            'total_clusters': len(self.clusters),
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_observations': self.total_observations,
            'num_clusters': len(self.clusters),
            'clusters': [
                {
                    'label': c.label,
                    'count': c.count,
                    'avg_latency_ms': c.avg_latency_ms,
                    'centroid': c.centroid.tolist(),
                }
                for c in self.clusters
            ],
        }


# ---------------------------------------------------------------------------
# Memory/Compute Tradeoff Optimizer
# ---------------------------------------------------------------------------

class TradeoffOptimizer:
    """
    Automatic memory/compute tradeoff optimization.

    Given a fixed resource budget, finds the Pareto-optimal
    configuration balancing memory usage against compute latency.
    Uses observed performance data to build a tradeoff curve.
    """

    def __init__(self, memory_budget_mb: float = 1024.0,
                 latency_budget_ms: float = 5.0):
        self.memory_budget_mb = memory_budget_mb
        self.latency_budget_ms = latency_budget_ms
        self.observations: List[Tuple[float, float, Dict[str, Any]]] = []
        # (memory_mb, latency_ms, config)

    def observe(self, memory_mb: float, latency_ms: float,
                config: Dict[str, Any]) -> None:
        """Record an observed (memory, latency, config) point"""
        self.observations.append((memory_mb, latency_ms, config))

    def pareto_front(self) -> List[Tuple[float, float, Dict[str, Any]]]:
        """
        Compute the Pareto front of observed (memory, latency) points.

        A point is Pareto-optimal if no other point is strictly better
        in both dimensions.
        """
        if not self.observations:
            return []

        sorted_obs = sorted(self.observations, key=lambda x: x[0])
        front = []
        best_latency = float('inf')

        for mem, lat, cfg in sorted_obs:
            if lat < best_latency:
                front.append((mem, lat, cfg))
                best_latency = lat

        return front

    def recommend(self) -> Optional[Dict[str, Any]]:
        """
        Recommend the best configuration within budget.

        Among Pareto-optimal points within both budgets, picks the
        one with the lowest latency.
        """
        front = self.pareto_front()
        if not front:
            return None

        feasible = [
            (mem, lat, cfg) for mem, lat, cfg in front
            if mem <= self.memory_budget_mb and lat <= self.latency_budget_ms
        ]

        if not feasible:
            # Relax: return the closest to budget
            return min(front, key=lambda x: x[1])[2]

        return min(feasible, key=lambda x: x[1])[2]

    def get_stats(self) -> Dict[str, Any]:
        front = self.pareto_front()
        return {
            'total_observations': len(self.observations),
            'pareto_front_size': len(front),
            'memory_budget_mb': self.memory_budget_mb,
            'latency_budget_ms': self.latency_budget_ms,
        }


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def create_bayesian_tuner(
        bounds: Dict[str, Tuple[float, float]],
        exploration_weight: float = 0.1) -> BayesianTuner:
    """Create a Bayesian-style hyperparameter tuner"""
    return BayesianTuner(bounds=bounds, exploration_weight=exploration_weight)


def create_workload_clusterer(distance_threshold: float = 1.5) -> WorkloadClusterer:
    """Create an online workload clusterer"""
    return WorkloadClusterer(distance_threshold=distance_threshold)


def create_tradeoff_optimizer(memory_budget_mb: float = 1024.0,
                              latency_budget_ms: float = 5.0) -> TradeoffOptimizer:
    """Create a memory/compute tradeoff optimizer"""
    return TradeoffOptimizer(
        memory_budget_mb=memory_budget_mb,
        latency_budget_ms=latency_budget_ms,
    )
