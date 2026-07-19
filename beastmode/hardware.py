#!/usr/bin/env python3
"""
BEASTMODE Hardware Feature Detection
=====================================

Comprehensive runtime hardware capability detection for optimal
backend and kernel selection.

Features:
- Real CPU feature detection (AVX2, AVX-512 variants, FMA, AMX, NEON)
- Cache topology discovery
- NUMA node detection
- Automatic backend recommendation with fallback paths
"""

import logging
import os
import platform
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger(__name__)

# CPU flags of interest, in rough order of SIMD capability
_X86_SIMD_FLAGS = [
    'sse4_1', 'sse4_2', 'avx', 'avx2', 'fma',
    'avx512f', 'avx512dq', 'avx512bw', 'avx512vl', 'avx512vnni',
    'amx_tile', 'amx_int8', 'amx_bf16',
]

_ARM_SIMD_FLAGS = ['neon', 'asimd', 'sve', 'sve2']


@dataclass
class CPUFeatures:
    """Detected CPU features and capabilities"""
    machine: str
    features: List[str] = field(default_factory=list)
    cpu_count: int = 1
    physical_cores: Optional[int] = None
    cache_line_size: int = 64
    l1_cache_kb: Optional[int] = None
    l2_cache_kb: Optional[int] = None
    l3_cache_kb: Optional[int] = None
    numa_nodes: int = 1

    @property
    def has_avx512(self) -> bool:
        return 'avx512f' in self.features

    @property
    def has_avx2(self) -> bool:
        return 'avx2' in self.features

    @property
    def has_fma(self) -> bool:
        return 'fma' in self.features

    @property
    def has_amx(self) -> bool:
        return 'amx_tile' in self.features

    @property
    def has_neon(self) -> bool:
        return 'neon' in self.features or 'asimd' in self.features

    @property
    def simd_vector_width(self) -> int:
        """Optimal SIMD vector width in float32 elements"""
        if self.has_avx512:
            return 16  # 512-bit
        if self.has_avx2:
            return 8   # 256-bit
        if 'sse4_1' in self.features or self.has_neon:
            return 4   # 128-bit
        return 1

    @property
    def optimal_alignment(self) -> int:
        """Optimal memory alignment in bytes for SIMD operations"""
        if self.has_avx512:
            return 64
        if self.has_avx2:
            return 32
        return 16

    def to_dict(self) -> Dict[str, Any]:
        return {
            'machine': self.machine,
            'features': self.features,
            'cpu_count': self.cpu_count,
            'physical_cores': self.physical_cores,
            'cache_line_size': self.cache_line_size,
            'l1_cache_kb': self.l1_cache_kb,
            'l2_cache_kb': self.l2_cache_kb,
            'l3_cache_kb': self.l3_cache_kb,
            'numa_nodes': self.numa_nodes,
            'has_avx512': self.has_avx512,
            'has_avx2': self.has_avx2,
            'has_fma': self.has_fma,
            'has_amx': self.has_amx,
            'simd_vector_width': self.simd_vector_width,
            'optimal_alignment': self.optimal_alignment,
        }


def _parse_proc_cpuinfo() -> List[str]:
    """Parse /proc/cpuinfo flags (Linux)"""
    flags: List[str] = []
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                lower = line.lower()
                if lower.startswith(('flags', 'features')):
                    raw = line.split(':', 1)[1].split()
                    candidates = _X86_SIMD_FLAGS + _ARM_SIMD_FLAGS
                    flags = [fl for fl in candidates if fl in raw]
                    break
    except (OSError, IndexError):
        pass
    return flags


def _read_int_file(path: str) -> Optional[int]:
    """Read an integer from a sysfs file, returns None on failure"""
    try:
        with open(path, 'r') as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _read_cache_size_kb(index: int) -> Optional[int]:
    """Read cache size in KB from sysfs for cpu0 cache index"""
    path = f'/sys/devices/system/cpu/cpu0/cache/index{index}/size'
    try:
        with open(path, 'r') as f:
            size = f.read().strip()
        if size.endswith('K'):
            return int(size[:-1])
        if size.endswith('M'):
            return int(size[:-1]) * 1024
        return int(size) // 1024
    except (OSError, ValueError):
        return None


def _detect_numa_nodes() -> int:
    """Count NUMA nodes via sysfs"""
    try:
        nodes = [d for d in os.listdir('/sys/devices/system/node')
                 if d.startswith('node') and d[4:].isdigit()]
        return max(len(nodes), 1)
    except OSError:
        return 1


@lru_cache(maxsize=1)
def detect_cpu_features() -> CPUFeatures:
    """
    Detect CPU features at runtime. Result is cached.

    Uses /proc/cpuinfo and sysfs on Linux; falls back to
    conservative defaults on other platforms.
    """
    machine = platform.machine()
    features = _parse_proc_cpuinfo()

    physical_cores = None
    try:
        import psutil
        physical_cores = psutil.cpu_count(logical=False)
    except ImportError:
        pass

    cache_line = _read_int_file(
        '/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size') or 64

    cpu_features = CPUFeatures(
        machine=machine,
        features=features,
        cpu_count=os.cpu_count() or 1,
        physical_cores=physical_cores,
        cache_line_size=cache_line,
        l1_cache_kb=_read_cache_size_kb(0),
        l2_cache_kb=_read_cache_size_kb(2),
        l3_cache_kb=_read_cache_size_kb(3),
        numa_nodes=_detect_numa_nodes(),
    )

    logger.info(f"CPU features detected: {machine}, "
                f"SIMD width={cpu_features.simd_vector_width}, "
                f"AVX-512={cpu_features.has_avx512}, FMA={cpu_features.has_fma}, "
                f"NUMA nodes={cpu_features.numa_nodes}")

    return cpu_features


def detect_gpu_capabilities() -> Dict[str, Any]:
    """
    Detect GPU capabilities at runtime.

    Returns availability info for CUDA and OpenCL backends
    without requiring the libraries to be installed.
    """
    capabilities = {
        'cuda_available': False,
        'opencl_available': False,
        'devices': [],
    }

    # CUDA detection via driver presence
    cuda_paths = ['/usr/local/cuda', '/opt/cuda']
    if any(os.path.isdir(p) for p in cuda_paths) or os.path.exists('/proc/driver/nvidia/version'):
        capabilities['cuda_available'] = True

    try:
        import ctypes
        ctypes.CDLL('libOpenCL.so.1')
        capabilities['opencl_available'] = True
    except OSError:
        pass

    return capabilities


def recommend_backend() -> Dict[str, Any]:
    """
    Recommend the optimal compute backend based on detected hardware.

    Returns a recommendation with fallback chain for unsupported operations.
    """
    cpu = detect_cpu_features()
    gpu = detect_gpu_capabilities()

    fallback_chain = []
    if gpu['cuda_available']:
        fallback_chain.append('gpu_cuda')
    if gpu['opencl_available']:
        fallback_chain.append('gpu_opencl')
    if cpu.machine in ('x86_64', 'AMD64'):
        fallback_chain.append('cpu_x86_64')
    elif cpu.machine in ('aarch64', 'arm64'):
        fallback_chain.append('cpu_arm64')
    else:
        fallback_chain.append('cpu_x86_64')  # generic fallback

    return {
        'primary': fallback_chain[0],
        'fallback_chain': fallback_chain,
        'cpu_features': cpu.to_dict(),
        'gpu_capabilities': gpu,
    }
