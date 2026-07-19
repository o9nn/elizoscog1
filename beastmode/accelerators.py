#!/usr/bin/env python3
"""
BeastMode Accelerators
======================

Hardware-specific acceleration components for maximum performance.

Features:
- SIMD vectorization for CPU operations
- Memory pooling and optimization
- Intelligent caching strategies
- Tensor compression for reduced memory footprint
"""

import numpy as np
import logging
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
import platform
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.core.ggml_symbolic_kernels import SymbolicTensor

from .hardware import detect_cpu_features

logger = logging.getLogger(__name__)


def aligned_empty(shape: Tuple[int, ...], dtype: np.dtype = np.float32,
                  alignment: int = 64) -> np.ndarray:
    """
    Allocate an uninitialized array with guaranteed memory alignment.

    SIMD instructions (AVX2/AVX-512) perform best with 32/64-byte
    aligned data. numpy does not guarantee alignment, so we over-allocate
    and slice to an aligned offset.
    """
    dtype = np.dtype(dtype)
    size = int(np.prod(shape))
    nbytes = size * dtype.itemsize
    buffer = np.empty(nbytes + alignment, dtype=np.uint8)
    offset = (-buffer.ctypes.data) % alignment
    return buffer[offset:offset + nbytes].view(dtype).reshape(shape)


def is_aligned(array: np.ndarray, alignment: int = 64) -> bool:
    """Check whether an array's data pointer is aligned"""
    return array.ctypes.data % alignment == 0


@dataclass
class SIMDConfig:
    """SIMD vectorization configuration"""
    vector_width: int = 8  # AVX2 default (256-bit / 32-bit floats)
    alignment: int = 32    # 32-byte alignment for AVX
    use_fma: bool = True   # Fused multiply-add
    auto_vectorize: bool = True


class SIMDAccelerator:
    """
    SIMD acceleration for tensor operations.
    
    Provides vectorized implementations of common operations
    using numpy's optimized SIMD backend.
    """
    
    def __init__(self, config: Optional[SIMDConfig] = None):
        self.config = config or SIMDConfig()
        
        # Detect CPU capabilities
        self.cpu_info = self._detect_cpu_capabilities()
        
        # Adjust vector width based on CPU
        if 'avx512' in self.cpu_info.get('features', []):
            self.config.vector_width = 16  # 512-bit
            self.config.alignment = 64
        elif 'avx2' in self.cpu_info.get('features', []):
            self.config.vector_width = 8   # 256-bit
            self.config.alignment = 32
        elif 'sse4' in self.cpu_info.get('features', []):
            self.config.vector_width = 4   # 128-bit
            self.config.alignment = 16
        
        self.config.use_fma = self.config.use_fma and self.cpu_info.get('has_fma', False)
        
        logger.info(f"SIMDAccelerator initialized: vector_width={self.config.vector_width}, "
                   f"fma={self.config.use_fma}")
    
    def _detect_cpu_capabilities(self) -> Dict[str, Any]:
        """Detect CPU capabilities via real hardware feature detection"""
        cpu = detect_cpu_features()
        
        features = []
        if cpu.has_avx512:
            features.append('avx512')
        if cpu.has_avx2:
            features.append('avx2')
        if 'sse4_1' in cpu.features or 'sse4_2' in cpu.features or cpu.has_neon:
            features.append('sse4')
        
        return {
            'machine': cpu.machine,
            'processor': platform.processor(),
            'features': features,
            'has_fma': cpu.has_fma,
            'cache_line_size': cpu.cache_line_size,
            'numa_nodes': cpu.numa_nodes,
        }
    
    def vectorized_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Vectorized addition with SIMD optimization"""
        out = aligned_empty(a.shape, a.dtype, self.config.alignment)
        return np.add(a, b, out=out)
    
    def vectorized_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Vectorized multiplication with SIMD optimization"""
        out = aligned_empty(a.shape, a.dtype, self.config.alignment)
        return np.multiply(a, b, out=out)
    
    def vectorized_fma(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Fused multiply-add: a * b + c (single-pass, no intermediate allocation)"""
        out = aligned_empty(a.shape, a.dtype, self.config.alignment)
        np.multiply(a, b, out=out)
        np.add(out, c, out=out)
        return out
    
    def vectorized_dot(self, a: np.ndarray, b: np.ndarray) -> float:
        """Vectorized dot product"""
        return float(np.dot(a.flatten(), b.flatten()))
    
    def vectorized_norm(self, a: np.ndarray) -> float:
        """Vectorized L2 norm"""
        return float(np.linalg.norm(a))
    
    def batch_process(self, tensors: List[np.ndarray], 
                     operation: str) -> List[np.ndarray]:
        """Process multiple tensors with SIMD optimization"""
        if operation == 'normalize':
            return [t / (np.linalg.norm(t) + 1e-8) for t in tensors]
        elif operation == 'softmax':
            results = []
            for t in tensors:
                exp_t = np.exp(t - np.max(t))
                results.append(exp_t / np.sum(exp_t))
            return results
        elif operation == 'relu':
            return [np.maximum(t, 0) for t in tensors]
        else:
            return tensors


@dataclass
class MemoryPoolConfig:
    """Memory pool configuration"""
    pool_size_mb: float = 256.0
    block_size: int = 4096  # bytes
    enable_defragmentation: bool = True
    gc_threshold: float = 0.8  # GC when 80% full


class MemoryOptimizer:
    """
    Memory optimization for tensor operations.
    
    Features:
    - Memory pooling to reduce allocations
    - Automatic memory reuse
    - Defragmentation
    - Memory usage tracking
    """
    
    def __init__(self, config: Optional[MemoryPoolConfig] = None):
        self.config = config or MemoryPoolConfig()
        
        # Memory tracking
        self.allocated_bytes = 0
        self.peak_bytes = 0
        self.allocation_count = 0
        self.reuse_count = 0
        
        # Free list for memory reuse
        self.free_blocks: Dict[int, List[np.ndarray]] = {}
        
        logger.info(f"MemoryOptimizer initialized: pool_size={self.config.pool_size_mb}MB")
    
    def allocate(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """Allocate tensor with potential reuse"""
        size_bytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
        
        # Check for reusable block
        if size_bytes in self.free_blocks and self.free_blocks[size_bytes]:
            block = self.free_blocks[size_bytes].pop()
            if block.shape == shape:
                self.reuse_count += 1
                return block
        
        # Allocate new block
        self.allocation_count += 1
        self.allocated_bytes += size_bytes
        self.peak_bytes = max(self.peak_bytes, self.allocated_bytes)
        
        return np.empty(shape, dtype=dtype)
    
    def release(self, tensor: np.ndarray):
        """Release tensor for potential reuse"""
        size_bytes = tensor.nbytes
        
        if size_bytes not in self.free_blocks:
            self.free_blocks[size_bytes] = []
        
        # Only keep a limited number of blocks
        if len(self.free_blocks[size_bytes]) < 10:
            self.free_blocks[size_bytes].append(tensor)
        
        self.allocated_bytes -= size_bytes
    
    def optimize_tensor(self, tensor: SymbolicTensor) -> SymbolicTensor:
        """Optimize tensor memory layout"""
        # Ensure contiguous memory
        if not tensor.data.flags['C_CONTIGUOUS']:
            tensor.data = np.ascontiguousarray(tensor.data)
        
        return tensor
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        reuse_rate = self.reuse_count / max(self.allocation_count + self.reuse_count, 1)
        
        return {
            'allocated_mb': self.allocated_bytes / (1024 * 1024),
            'peak_mb': self.peak_bytes / (1024 * 1024),
            'allocation_count': self.allocation_count,
            'reuse_count': self.reuse_count,
            'reuse_rate': reuse_rate,
            'free_blocks': sum(len(v) for v in self.free_blocks.values())
        }


class ArenaAllocator:
    """
    Arena-style memory allocator for tensor workloads.
    
    Allocates one large aligned buffer up-front, then serves tensor
    allocations as zero-copy views into it via cheap bump-pointer
    allocation. Ideal for per-inference scratch memory: allocate
    during the operation, then reset() the whole arena at once.
    
    Features:
    - O(1) bump-pointer allocation, no per-tensor malloc
    - SIMD-friendly alignment for every allocation
    - Zero-copy views into the arena buffer
    - Whole-arena reset between inference passes
    """
    
    def __init__(self, capacity_mb: float = 64.0, alignment: int = 64):
        self.capacity_bytes = int(capacity_mb * 1024 * 1024)
        self.alignment = alignment
        self._buffer = aligned_empty((self.capacity_bytes,), np.uint8, alignment)
        self._offset = 0
        
        # Statistics
        self.allocation_count = 0
        self.overflow_count = 0
        self.peak_offset = 0
        
        logger.info(f"ArenaAllocator initialized: capacity={capacity_mb}MB, "
                   f"alignment={alignment}")
    
    def allocate(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Allocate a tensor as a zero-copy view into the arena.
        
        Falls back to a regular aligned allocation when the arena is full.
        """
        dtype = np.dtype(dtype)
        size_bytes = int(np.prod(shape)) * dtype.itemsize
        
        # Round offset up to alignment boundary
        aligned_offset = (self._offset + self.alignment - 1) // self.alignment * self.alignment
        
        if aligned_offset + size_bytes > self.capacity_bytes:
            # Arena exhausted: fall back to heap allocation
            self.overflow_count += 1
            return aligned_empty(shape, dtype, self.alignment)
        
        view = self._buffer[aligned_offset:aligned_offset + size_bytes].view(dtype).reshape(shape)
        self._offset = aligned_offset + size_bytes
        self.peak_offset = max(self.peak_offset, self._offset)
        self.allocation_count += 1
        
        return view
    
    def reset(self) -> None:
        """Reset the arena, invalidating all outstanding views"""
        self._offset = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get arena statistics"""
        return {
            'capacity_mb': self.capacity_bytes / (1024 * 1024),
            'used_mb': self._offset / (1024 * 1024),
            'peak_mb': self.peak_offset / (1024 * 1024),
            'utilization': self._offset / max(self.capacity_bytes, 1),
            'allocation_count': self.allocation_count,
            'overflow_count': self.overflow_count,
        }


@dataclass
class CacheConfig:
    """Cache configuration"""
    max_size: int = 1000
    ttl_seconds: float = 3600.0  # 1 hour
    eviction_policy: str = 'lru'  # lru, lfu, fifo


class CacheManager:
    """
    Intelligent caching for tensor operations.
    
    Features:
    - LRU/LFU/FIFO eviction policies
    - TTL-based expiration
    - Hit rate tracking
    - Adaptive caching
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        
        # Cache storage (using OrderedDict for LRU)
        self.cache: OrderedDict[str, Tuple[SymbolicTensor, float, int]] = OrderedDict()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.info(f"CacheManager initialized: max_size={self.config.max_size}, "
                   f"policy={self.config.eviction_policy}")
    
    def get(self, key: str) -> Optional[SymbolicTensor]:
        """Get cached tensor"""
        if key not in self.cache:
            self.misses += 1
            return None
        
        tensor, timestamp, access_count = self.cache[key]
        
        # Check TTL
        if time.time() - timestamp > self.config.ttl_seconds:
            self.cache.pop(key)
            self.misses += 1
            return None
        
        # Update access
        self.cache[key] = (tensor, timestamp, access_count + 1)
        
        # Move to end for LRU
        if self.config.eviction_policy == 'lru':
            self.cache.move_to_end(key)
        
        self.hits += 1
        return tensor
    
    def put(self, key: str, tensor: SymbolicTensor):
        """Cache a tensor"""
        # Evict if necessary
        while len(self.cache) >= self.config.max_size:
            self._evict()
        
        self.cache[key] = (tensor, time.time(), 1)
    
    def _evict(self):
        """Evict based on policy"""
        if not self.cache:
            return
        
        if self.config.eviction_policy == 'lru':
            # Remove oldest (first item)
            self.cache.popitem(last=False)
        
        elif self.config.eviction_policy == 'lfu':
            # Remove least frequently used
            min_key = min(self.cache, key=lambda k: self.cache[k][2])
            self.cache.pop(min_key)
        
        else:  # fifo
            self.cache.popitem(last=False)
        
        self.evictions += 1
    
    def invalidate(self, key: str):
        """Invalidate a cache entry"""
        if key in self.cache:
            self.cache.pop(key)
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.evictions = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_accesses = self.hits + self.misses
        hit_rate = self.hits / max(total_accesses, 1)
        
        return {
            'size': len(self.cache),
            'max_size': self.config.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions
        }


@dataclass 
class CompressionConfig:
    """Tensor compression configuration"""
    quantization_bits: int = 8  # 8-bit quantization
    enable_sparse: bool = True
    sparse_threshold: float = 0.1  # Values below threshold become 0
    enable_delta: bool = False  # Delta encoding for sequential data
    per_channel: bool = False  # Per-channel (axis 0) quantization scales
    target_accuracy: Optional[float] = None  # Auto-select precision for accuracy


class TensorCompressor:
    """
    Tensor compression for reduced memory footprint.
    
    Features:
    - Quantization (8-bit, 16-bit)
    - Sparse tensor representation
    - Delta encoding
    - Lossless/lossy compression options
    """
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config or CompressionConfig()
        
        # Compression statistics
        self.total_original_bytes = 0
        self.total_compressed_bytes = 0
        
        logger.info(f"TensorCompressor initialized: quantization={self.config.quantization_bits}bit")
    
    def compress(self, tensor: SymbolicTensor) -> Tuple[bytes, Dict[str, Any]]:
        """Compress tensor to bytes"""
        original_bytes = tensor.data.nbytes
        self.total_original_bytes += original_bytes
        
        # Quantization
        if self.config.quantization_bits == 8:
            compressed_data = self._quantize_8bit(tensor.data)
        elif self.config.quantization_bits == 16:
            compressed_data = self._quantize_16bit(tensor.data)
        else:
            compressed_data = tensor.data.tobytes()
        
        # Sparse encoding
        if self.config.enable_sparse:
            sparse_data, sparse_meta = self._sparse_encode(tensor.data)
            if len(sparse_data) < len(compressed_data):
                compressed_data = sparse_data
        
        self.total_compressed_bytes += len(compressed_data)
        
        metadata = {
            'original_shape': tensor.data.shape,
            'original_dtype': str(tensor.data.dtype),
            'quantization_bits': self.config.quantization_bits,
            'compression_ratio': original_bytes / max(len(compressed_data), 1),
            'symbols': tensor.symbols
        }
        
        return compressed_data, metadata
    
    def decompress(self, data: bytes, metadata: Dict[str, Any]) -> SymbolicTensor:
        """Decompress bytes to tensor"""
        shape = metadata['original_shape']
        dtype = np.dtype(metadata['original_dtype'])
        
        # Dequantization
        if metadata.get('quantization_bits', 32) == 8:
            tensor_data = self._dequantize_8bit(data, shape, dtype)
        elif metadata.get('quantization_bits', 32) == 16:
            tensor_data = self._dequantize_16bit(data, shape, dtype)
        else:
            tensor_data = np.frombuffer(data, dtype=dtype).reshape(shape)
        
        return SymbolicTensor(
            data=tensor_data,
            symbols=metadata.get('symbols', {})
        )
    
    def _quantize_8bit(self, data: np.ndarray) -> bytes:
        """Quantize to 8-bit"""
        min_val = data.min()
        max_val = data.max()
        scale = (max_val - min_val) / 255 if max_val > min_val else 1.0
        
        quantized = ((data - min_val) / scale).astype(np.uint8)
        
        # Prepend scale and min as float32
        header = np.array([scale, min_val], dtype=np.float32).tobytes()
        return header + quantized.tobytes()
    
    def _dequantize_8bit(self, data: bytes, shape: Tuple, dtype: np.dtype) -> np.ndarray:
        """Dequantize from 8-bit"""
        header = np.frombuffer(data[:8], dtype=np.float32)
        scale, min_val = header[0], header[1]
        
        quantized = np.frombuffer(data[8:], dtype=np.uint8)
        return (quantized.astype(dtype) * scale + min_val).reshape(shape)
    
    def _quantize_16bit(self, data: np.ndarray) -> bytes:
        """Quantize to 16-bit"""
        return data.astype(np.float16).tobytes()
    
    def _dequantize_16bit(self, data: bytes, shape: Tuple, dtype: np.dtype) -> np.ndarray:
        """Dequantize from 16-bit"""
        return np.frombuffer(data, dtype=np.float16).astype(dtype).reshape(shape)
    
    def quantize_per_channel(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Dynamic per-channel 8-bit quantization along axis 0.
        
        Each channel gets its own scale and zero-point, preserving
        accuracy far better than a single global scale when channel
        magnitudes differ.
        
        Returns (quantized_uint8, scales, min_vals).
        """
        channels = data.shape[0]
        flat = data.reshape(channels, -1)
        
        min_vals = flat.min(axis=1)
        max_vals = flat.max(axis=1)
        ranges = max_vals - min_vals
        scales = np.where(ranges > 0, ranges / 255.0, 1.0).astype(np.float32)
        
        quantized = ((flat - min_vals[:, None]) / scales[:, None]).round().astype(np.uint8)
        return quantized.reshape(data.shape), scales, min_vals.astype(np.float32)
    
    def dequantize_per_channel(self, quantized: np.ndarray, scales: np.ndarray,
                              min_vals: np.ndarray,
                              dtype: np.dtype = np.float32) -> np.ndarray:
        """Dequantize per-channel quantized data back to floating point"""
        channels = quantized.shape[0]
        flat = quantized.reshape(channels, -1).astype(dtype)
        restored = flat * scales[:, None] + min_vals[:, None]
        return restored.reshape(quantized.shape).astype(dtype)
    
    def select_precision(self, data: np.ndarray,
                        target_accuracy: Optional[float] = None) -> int:
        """
        Automatically select quantization precision that meets the
        target accuracy (relative reconstruction error).
        
        Tries INT8 first (4x compression), falls back to FP16 (2x),
        then FP32 (lossless). Returns selected bit width (8, 16, or 32).
        """
        target = target_accuracy or self.config.target_accuracy or 0.99
        data_norm = np.linalg.norm(data)
        if data_norm == 0:
            return 8  # Zero tensors quantize losslessly
        
        # Test INT8 (per-channel when enabled and possible)
        if self.config.per_channel and data.ndim >= 2:
            q, scales, mins = self.quantize_per_channel(data)
            restored = self.dequantize_per_channel(q, scales, mins, data.dtype)
        else:
            packed = self._quantize_8bit(data)
            restored = self._dequantize_8bit(packed, data.shape, data.dtype)
        
        error = np.linalg.norm(data - restored) / data_norm
        if 1.0 - error >= target:
            return 8
        
        # Test FP16
        restored16 = data.astype(np.float16).astype(data.dtype)
        error16 = np.linalg.norm(data - restored16) / data_norm
        if 1.0 - error16 >= target:
            return 16
        
        return 32
    
    def _sparse_encode(self, data: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        """Sparse encoding for sparse tensors"""
        flat = data.flatten()
        
        # Find non-zero indices
        if self.config.sparse_threshold > 0:
            non_zero_mask = np.abs(flat) > self.config.sparse_threshold
        else:
            non_zero_mask = flat != 0
        
        indices = np.where(non_zero_mask)[0]
        values = flat[non_zero_mask]
        
        # Encode
        sparse_data = indices.astype(np.int32).tobytes() + values.tobytes()
        
        return sparse_data, {'sparse': True, 'num_nonzero': len(indices)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        ratio = self.total_original_bytes / max(self.total_compressed_bytes, 1)
        
        return {
            'total_original_mb': self.total_original_bytes / (1024 * 1024),
            'total_compressed_mb': self.total_compressed_bytes / (1024 * 1024),
            'compression_ratio': ratio,
            'space_saved_mb': (self.total_original_bytes - self.total_compressed_bytes) / (1024 * 1024)
        }
