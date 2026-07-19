#!/usr/bin/env python3
"""
BEASTMODE Kernel Fusion
========================

Fuses sequences of tensor operations into single-pass kernels,
eliminating intermediate tensor materialization and improving
cache locality.

Features:
- Pre-built fused kernels for common sequences (matmul+bias+activation, etc.)
- Operation graph analysis for automatic fusion opportunities
- Single-allocation execution with in-place intermediate updates
"""

import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fused elementwise kernels (single pass, in-place where possible)
# ---------------------------------------------------------------------------

def fused_multiply_add(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """a * b + c without intermediate allocation"""
    out = np.multiply(a, b)
    np.add(out, c, out=out)
    return out


def fused_matmul_bias(x: np.ndarray, w: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """x @ w + bias in a fused pass"""
    out = np.matmul(x, w)
    np.add(out, bias, out=out)
    return out


def fused_matmul_bias_relu(x: np.ndarray, w: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """x @ w + bias followed by ReLU, all in-place after the matmul"""
    out = np.matmul(x, w)
    np.add(out, bias, out=out)
    np.maximum(out, 0, out=out)
    return out


def fused_normalize_relu(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize then ReLU in a single fused pass"""
    norm = np.linalg.norm(x) + eps
    out = np.divide(x, norm)
    np.maximum(out, 0, out=out)
    return out


def fused_scale_shift_clip(x: np.ndarray, scale: float, shift: float,
                          min_val: float, max_val: float) -> np.ndarray:
    """(x * scale + shift) clipped to [min_val, max_val], single allocation"""
    out = np.multiply(x, scale)
    np.add(out, shift, out=out)
    np.clip(out, min_val, max_val, out=out)
    return out


def fused_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax with minimal allocations"""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    np.exp(shifted, out=shifted)
    denominator = np.sum(shifted, axis=axis, keepdims=True)
    np.divide(shifted, denominator, out=shifted)
    return shifted


# Registry of fusable operation sequences -> fused kernel
FUSION_PATTERNS: Dict[Tuple[str, ...], Callable] = {
    ('multiply', 'add'): fused_multiply_add,
    ('matmul', 'add'): fused_matmul_bias,
    ('matmul', 'add', 'relu'): fused_matmul_bias_relu,
    ('normalize', 'relu'): fused_normalize_relu,
    ('sub_max', 'exp', 'div_sum'): fused_softmax,
}


@dataclass
class FusionStats:
    """Statistics for fusion pipeline execution"""
    operations_seen: int = 0
    operations_fused: int = 0
    fusion_groups: int = 0
    total_time_ms: float = 0.0

    @property
    def fusion_rate(self) -> float:
        return self.operations_fused / max(self.operations_seen, 1)


class OperationGraph:
    """
    Lightweight operation graph for fusion analysis.

    Operations are added as (name, inputs) nodes; the analyzer finds
    linear chains matching known fusion patterns.
    """

    def __init__(self):
        self.operations: List[Dict[str, Any]] = []

    def add_operation(self, name: str, **params) -> int:
        """Add an operation node, returns its index"""
        self.operations.append({'name': name, 'params': params})
        return len(self.operations) - 1

    def find_fusion_groups(self) -> List[Tuple[int, int, Callable]]:
        """
        Find maximal fusable subsequences.

        Returns list of (start_index, end_index_exclusive, fused_kernel),
        longest patterns matched first, non-overlapping, greedy left-to-right.
        """
        names = [op['name'] for op in self.operations]
        groups: List[Tuple[int, int, Callable]] = []

        # Sort patterns by length descending for maximal fusion
        patterns = sorted(FUSION_PATTERNS.items(), key=lambda kv: -len(kv[0]))

        i = 0
        while i < len(names):
            matched = False
            for pattern, kernel in patterns:
                plen = len(pattern)
                if tuple(names[i:i + plen]) == pattern:
                    groups.append((i, i + plen, kernel))
                    i += plen
                    matched = True
                    break
            if not matched:
                i += 1

        return groups


class FusionPipeline:
    """
    Executes sequences of elementwise operations with automatic fusion.

    Analyzes the requested operation sequence, replaces fusable
    subsequences with fused single-pass kernels, and executes the rest
    with in-place fallbacks.
    """

    # Fallback single-op kernels (in-place where valid)
    _SINGLE_OPS: Dict[str, Callable[..., np.ndarray]] = {
        'relu': lambda x: np.maximum(x, 0),
        'add': lambda x, other: np.add(x, other),
        'multiply': lambda x, other: np.multiply(x, other),
        'matmul': lambda x, other: np.matmul(x, other),
        'normalize': lambda x: x / (np.linalg.norm(x) + 1e-8),
        'softmax': fused_softmax,
    }

    def __init__(self):
        self.stats = FusionStats()
        logger.info("FusionPipeline initialized with "
                    f"{len(FUSION_PATTERNS)} fusion patterns")

    def execute(self, x: np.ndarray,
                operations: List[Tuple[str, Dict[str, Any]]]) -> np.ndarray:
        """
        Execute a sequence of operations on x with automatic fusion.

        Args:
            x: input tensor
            operations: list of (op_name, params) tuples. Params may
                contain 'other' (second operand) or kernel-specific args.

        Returns:
            Result tensor.
        """
        start = time.perf_counter()

        graph = OperationGraph()
        for name, params in operations:
            graph.add_operation(name, **params)

        groups = graph.find_fusion_groups()
        fused_ranges = {g[0]: g for g in groups}
        fused_indices = set()
        for start_idx, end_idx, _ in groups:
            fused_indices.update(range(start_idx, end_idx))

        self.stats.operations_seen += len(operations)
        self.stats.fusion_groups += len(groups)

        result = x
        i = 0
        while i < len(operations):
            if i in fused_ranges:
                start_idx, end_idx, kernel = fused_ranges[i]
                result = self._execute_fused(result, kernel,
                                             operations[start_idx:end_idx])
                self.stats.operations_fused += end_idx - start_idx
                i = end_idx
            else:
                name, params = operations[i]
                result = self._execute_single(result, name, params)
                i += 1

        self.stats.total_time_ms += (time.perf_counter() - start) * 1000
        return result

    def _execute_fused(self, x: np.ndarray, kernel: Callable,
                       ops: List[Tuple[str, Dict[str, Any]]]) -> np.ndarray:
        """Execute a fused kernel, collecting operands from the ops"""
        if kernel is fused_multiply_add:
            return kernel(x, ops[0][1]['other'], ops[1][1]['other'])
        if kernel is fused_matmul_bias:
            return kernel(x, ops[0][1]['other'], ops[1][1]['other'])
        if kernel is fused_matmul_bias_relu:
            return kernel(x, ops[0][1]['other'], ops[1][1]['other'])
        if kernel is fused_normalize_relu:
            return kernel(x)
        if kernel is fused_softmax:
            return kernel(x)
        raise ValueError(f"Unknown fused kernel: {kernel}")

    def _execute_single(self, x: np.ndarray, name: str,
                        params: Dict[str, Any]) -> np.ndarray:
        """Execute a single non-fused operation"""
        if name not in self._SINGLE_OPS:
            raise ValueError(f"Unknown operation: {name}")
        op = self._SINGLE_OPS[name]
        if 'other' in params:
            return op(x, params['other'])
        return op(x)

    def get_stats(self) -> Dict[str, Any]:
        """Get fusion pipeline statistics"""
        return {
            'operations_seen': self.stats.operations_seen,
            'operations_fused': self.stats.operations_fused,
            'fusion_groups': self.stats.fusion_groups,
            'fusion_rate': self.stats.fusion_rate,
            'total_time_ms': self.stats.total_time_ms,
        }


def create_fusion_pipeline() -> FusionPipeline:
    """Factory function to create a fusion pipeline"""
    return FusionPipeline()
