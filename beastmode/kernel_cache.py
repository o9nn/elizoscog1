#!/usr/bin/env python3
"""
BEASTMODE Predictive Kernel Cache
==================================

Persistent kernel-profile cache with ahead-of-time (AOT) warmup and
kernel specialization based on input characteristics.

Features:
- File-backed kernel profile persistence (JSON)
- AOT warmup: pre-load and pre-compile common operation patterns
- Kernel specialization keyed on (operation, shape-class, architecture)
- Adaptive re-profiling when stale
"""

import json
import logging
import os
import time
import hashlib
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from .hardware import detect_cpu_features

logger = logging.getLogger(__name__)


@dataclass
class SpecializedKernelProfile:
    """Performance profile for a specialized kernel instance"""
    operation: str
    shape_class: str          # e.g. "small_dense", "large_dense"
    architecture: str
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    throughput_ops_sec: float = 0.0
    execution_count: int = 0
    last_updated: float = field(default_factory=time.time)

    @property
    def is_stale(self) -> bool:
        """Profile is stale after 1 hour without updates"""
        return (time.time() - self.last_updated) > 3600.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation': self.operation,
            'shape_class': self.shape_class,
            'architecture': self.architecture,
            'avg_latency_ms': self.avg_latency_ms,
            'min_latency_ms': self.min_latency_ms if self.min_latency_ms != float('inf') else 0.0,
            'max_latency_ms': self.max_latency_ms,
            'throughput_ops_sec': self.throughput_ops_sec,
            'execution_count': self.execution_count,
            'last_updated': self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SpecializedKernelProfile':
        return cls(
            operation=d['operation'],
            shape_class=d['shape_class'],
            architecture=d['architecture'],
            avg_latency_ms=d.get('avg_latency_ms', 0.0),
            min_latency_ms=d.get('min_latency_ms', 0.0),
            max_latency_ms=d.get('max_latency_ms', 0.0),
            throughput_ops_sec=d.get('throughput_ops_sec', 0.0),
            execution_count=d.get('execution_count', 0),
            last_updated=d.get('last_updated', time.time()),
        )


def classify_shape(shape: Tuple[int, ...]) -> str:
    """
    Classify a tensor shape into a size class for kernel specialization.

    Classes: small_dense (<1K), medium_dense (1K-100K), large_dense (>100K)
    """
    total = int(np.prod(shape)) if shape else 0
    if total < 1_000:
        return 'small_dense'
    if total < 100_000:
        return 'medium_dense'
    return 'large_dense'


class KernelProfileCache:
    """
    Persistent cache of kernel performance profiles.

    Stores specialized profiles keyed by (operation, shape_class, architecture)
    and persists them to a JSON file for cross-session reuse (AOT warmup).

    Usage:
        cache = KernelProfileCache()
        cache.load()
        profile = cache.get('PATTERN_RECOGNITION', 'medium_dense', 'cpu_x86_64')
        cache.update('PATTERN_RECOGNITION', 'medium_dense', 'cpu_x86_64', 1.2)
        cache.save()
    """

    def __init__(self, cache_path: Optional[str] = None):
        if cache_path is None:
            cache_dir = os.path.join(tempfile.gettempdir(), 'beastmode_cache')
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, 'kernel_profiles.json')

        self.cache_path = cache_path
        self._profiles: Dict[str, SpecializedKernelProfile] = {}
        self._dirty = False

        # JIT warmup state
        self._warmed_up = False
        self._warmup_patterns: List[Tuple[str, str]] = []

        logger.info(f"KernelProfileCache: path={cache_path}")

    @staticmethod
    def _key(operation: str, shape_class: str, architecture: str) -> str:
        return f"{operation}::{shape_class}::{architecture}"

    def get(self, operation: str, shape_class: str,
            architecture: str) -> Optional[SpecializedKernelProfile]:
        """Get a cached profile, returning None if not found or stale"""
        key = self._key(operation, shape_class, architecture)
        profile = self._profiles.get(key)
        if profile is not None and profile.is_stale:
            logger.debug(f"Profile {key} is stale, will re-profile")
            return None
        return profile

    def update(self, operation: str, shape_class: str,
               architecture: str, latency_ms: float,
               learning_rate: float = 0.1) -> SpecializedKernelProfile:
        """
        Update a profile with a new latency observation.

        Uses exponential moving average for smooth adaptation.
        """
        key = self._key(operation, shape_class, architecture)

        if key in self._profiles:
            p = self._profiles[key]
            alpha = learning_rate
            p.avg_latency_ms = alpha * latency_ms + (1 - alpha) * p.avg_latency_ms
            p.min_latency_ms = min(p.min_latency_ms, latency_ms)
            p.max_latency_ms = max(p.max_latency_ms, latency_ms)
            p.throughput_ops_sec = 1000.0 / max(p.avg_latency_ms, 0.001)
            p.execution_count += 1
            p.last_updated = time.time()
        else:
            p = SpecializedKernelProfile(
                operation=operation,
                shape_class=shape_class,
                architecture=architecture,
                avg_latency_ms=latency_ms,
                min_latency_ms=latency_ms,
                max_latency_ms=latency_ms,
                throughput_ops_sec=1000.0 / max(latency_ms, 0.001),
                execution_count=1,
            )
            self._profiles[key] = p

        self._dirty = True
        return p

    def best_architecture(self, operation: str,
                          shape_class: str) -> Optional[str]:
        """
        Return the architecture with the lowest avg latency for a given
        operation and shape class, or None if no data exists.
        """
        best_arch = None
        best_latency = float('inf')

        for key, profile in self._profiles.items():
            if (profile.operation == operation
                    and profile.shape_class == shape_class
                    and not profile.is_stale
                    and profile.avg_latency_ms < best_latency):
                best_latency = profile.avg_latency_ms
                best_arch = profile.architecture

        return best_arch

    def save(self) -> bool:
        """Persist profiles to disk"""
        if not self._dirty:
            return False

        data = {
            'version': 1,
            'saved_at': time.time(),
            'cpu_fingerprint': self._cpu_fingerprint(),
            'profiles': {
                k: v.to_dict() for k, v in self._profiles.items()
            },
        }

        try:
            # Atomic write: write to temp file then rename
            tmp_path = self.cache_path + '.tmp'
            with open(tmp_path, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self.cache_path)
            self._dirty = False
            logger.info(f"Saved {len(self._profiles)} kernel profiles to {self.cache_path}")
            return True
        except OSError as e:
            logger.warning(f"Failed to save kernel profiles: {e}")
            return False

    def load(self) -> int:
        """
        Load profiles from disk. Returns number of profiles loaded.

        Skips profiles from a different CPU (different fingerprint)
        to avoid using misleading cross-machine data.
        """
        if not os.path.exists(self.cache_path):
            return 0

        try:
            with open(self.cache_path, 'r') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load kernel profiles: {e}")
            return 0

        # Check CPU compatibility
        saved_fp = data.get('cpu_fingerprint', '')
        current_fp = self._cpu_fingerprint()
        if saved_fp != current_fp:
            logger.info(
                f"CPU fingerprint changed ({saved_fp} -> {current_fp}), "
                "discarding cached profiles"
            )
            return 0

        count = 0
        for key, profile_dict in data.get('profiles', {}).items():
            try:
                self._profiles[key] = SpecializedKernelProfile.from_dict(profile_dict)
                count += 1
            except (KeyError, TypeError) as e:
                logger.debug(f"Skipping malformed profile {key}: {e}")

        logger.info(f"Loaded {count} kernel profiles from {self.cache_path}")
        return count

    def warmup(self, common_patterns: Optional[List[Tuple[str, str]]] = None) -> None:
        """
        Ahead-of-time warmup: pre-load profiles for common operation patterns.

        This eliminates cold-start latency for known workloads by ensuring
        profiles are in memory before the first real inference request.
        """
        if self._warmed_up:
            return

        self.load()

        if common_patterns:
            self._warmup_patterns = common_patterns

        # Pre-populate default patterns if none provided
        if not self._warmup_patterns:
            cpu = detect_cpu_features()
            self._warmup_patterns = [
                ('PATTERN_RECOGNITION', 'small_dense'),
                ('PATTERN_RECOGNITION', 'medium_dense'),
                ('TENSOR_TO_SYMBOL', 'small_dense'),
                ('CONTEXT_BINDING', 'medium_dense'),
            ]
            logger.debug(f"Warmup patterns registered for {cpu.machine}")

        self._warmed_up = True
        logger.info(
            f"AOT warmup complete: {len(self._profiles)} profiles loaded, "
            f"{len(self._warmup_patterns)} patterns registered"
        )

    @staticmethod
    def _cpu_fingerprint() -> str:
        """Generate a fingerprint of the current CPU for cache validation"""
        cpu = detect_cpu_features()
        raw = f"{cpu.machine}:{cpu.simd_vector_width}:{cpu.cpu_count}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    @property
    def profile_count(self) -> int:
        return len(self._profiles)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stale = sum(1 for p in self._profiles.values() if p.is_stale)
        return {
            'total_profiles': len(self._profiles),
            'stale_profiles': stale,
            'cache_path': self.cache_path,
            'warmed_up': self._warmed_up,
            'warmup_patterns': len(self._warmup_patterns),
        }


def create_kernel_cache(cache_path: Optional[str] = None) -> KernelProfileCache:
    """Factory: create and warm up a kernel profile cache"""
    cache = KernelProfileCache(cache_path=cache_path)
    cache.warmup()
    return cache
