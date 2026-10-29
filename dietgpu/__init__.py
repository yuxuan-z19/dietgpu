# Copyright (c) 2026 Yee Hin Chong (yuxuan-z)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
DietGPU: GPU-accelerated compression library for PyTorch.

This module provides efficient compression/decompression operations for tensors
on CUDA devices, supporting float16, bfloat16, float32, and arbitrary byte data.
"""

import os
from pathlib import Path

import torch
from torch import Tensor


# Load the C++ library
def _load_library() -> None:
    """Load the dietgpu C++ library."""
    # Try to find the library in common locations
    search_paths = []

    # 1. Environment variable (for custom builds)
    if "DIETGPU_BUILD_PATH" in os.environ:
        build_path = Path(os.environ["DIETGPU_BUILD_PATH"])
        search_paths.append(build_path / "lib/libdietgpu.so")

    # 2. Installed package location (when installed via pip/uv)
    current_dir = Path(__file__).parent
    search_paths.extend(
        [
            current_dir / "libdietgpu.so",  # Direct install in package dir
            current_dir / "lib/libdietgpu.so",  # Installed with lib subdirectory
        ]
    )

    # 3. Development build locations
    search_paths.extend(
        [
            current_dir / "build/lib/libdietgpu.so",
            current_dir.parent / "build/lib/libdietgpu.so",
            Path("build/lib/libdietgpu.so"),
        ]
    )

    # 4. scikit-build-core build directory (build/cp*/lib/)
    project_root = current_dir.parent
    search_paths.extend(project_root.glob("build/cp*/lib/libdietgpu.so"))

    # Try to load the library
    for lib_path in search_paths:
        if lib_path.exists():
            torch.ops.load_library(str(lib_path))
            return

    # If not found, raise an error with helpful message
    raise RuntimeError(
        "Cannot find libdietgpu.so. Please either:\n"
        "1. Install the package: pip install -e . (or uv pip install -e .)\n"
        "2. Build manually: mkdir build && cd build && cmake .. && make\n"
        "3. Set DIETGPU_BUILD_PATH environment variable"
    )


# Load library on import
_load_library()


# ============================================================================
# Utility Functions
# ============================================================================


def max_float_compressed_output_size(ts: list[Tensor]) -> tuple[int, int]:
    """
    Calculate the maximum compressed output size for float tensors.

    Args:
        ts: List of input tensors (must all have the same dtype)

    Returns:
        Tuple of (batch_size, max_compressed_size)
    """
    return torch.ops.dietgpu.max_float_compressed_output_size(ts)


def max_float_compressed_size(dtype: Tensor, size: int) -> int:
    """
    Calculate the maximum compressed size for a float tensor.

    Args:
        dtype: A tensor representing the dtype
        size: Number of elements

    Returns:
        Maximum compressed size in bytes
    """
    return torch.ops.dietgpu.max_float_compressed_size(dtype, size)


def max_any_compressed_output_size(ts: list[Tensor]) -> tuple[int, int]:
    """
    Calculate the maximum compressed output size for any tensor type.

    Args:
        ts: List of input tensors

    Returns:
        Tuple of (batch_size, max_compressed_size)
    """
    return torch.ops.dietgpu.max_any_compressed_output_size(ts)


def max_any_compressed_size(bytes: int) -> int:
    """
    Calculate the maximum compressed size for given bytes.

    Args:
        bytes: Number of bytes

    Returns:
        Maximum compressed size in bytes
    """
    return torch.ops.dietgpu.max_any_compressed_size(bytes)


# ============================================================================
# Compression Functions
# ============================================================================


def compress_data(
    compress_as_float: bool,
    ts_in: list[Tensor],
    checksum: bool = False,
    temp_mem: Tensor | None = None,
    out_compressed: Tensor | None = None,
    out_compressed_bytes: Tensor | None = None,
) -> tuple[Tensor, Tensor, int]:
    """
    Compress a batch of tensors.

    Args:
        compress_as_float: If True, compress as float data (requires float16, bfloat16, or float32)
        ts_in: List of input tensors to compress
        checksum: Whether to include checksum in compressed data
        temp_mem: Optional temporary memory buffer (uint8 tensor on CUDA)
        out_compressed: Optional pre-allocated output buffer for compressed data
        out_compressed_bytes: Optional pre-allocated output buffer for compressed sizes

    Returns:
        Tuple of:
            - compressed_data: 2D uint8 tensor of shape [batch_size, max_compressed_size]
            - compressed_sizes: 1D int32 tensor of shape [batch_size] with actual compressed sizes
            - temp_mem_usage: Amount of temporary memory used in bytes

    Example:
        >>> import torch
        >>> import dietgpu
        >>> dev = torch.device("cuda:0")
        >>> tensors = [torch.randn(1000, dtype=torch.float16, device=dev) for _ in range(3)]
        >>> comp, sizes, mem_usage = dietgpu.compress_data(True, tensors, checksum=True)
    """
    return torch.ops.dietgpu.compress_data(
        compress_as_float,
        ts_in,
        checksum,
        temp_mem,
        out_compressed,
        out_compressed_bytes,
    )


def compress_data_split_size(
    compress_as_float: bool,
    t_in: Tensor,
    t_in_split_sizes: Tensor,
    checksum: bool = False,
    temp_mem: Tensor | None = None,
    out_compressed: Tensor | None = None,
    out_compressed_bytes: Tensor | None = None,
) -> tuple[list[Tensor], Tensor, int]:
    """
    Compress a single tensor by splitting it into multiple parts.

    Args:
        compress_as_float: If True, compress as float data
        t_in: Input tensor to compress
        t_in_split_sizes: 1D int32 tensor on CPU with sizes for each split
        checksum: Whether to include checksum in compressed data
        temp_mem: Optional temporary memory buffer (uint8 tensor on CUDA)
        out_compressed: Optional pre-allocated output buffer for compressed data
        out_compressed_bytes: Optional pre-allocated output buffer for compressed sizes

    Returns:
        Tuple of:
            - compressed_data_list: List of 1D uint8 tensors, one per split
            - compressed_sizes: 1D int32 tensor with actual compressed sizes
            - temp_mem_usage: Amount of temporary memory used in bytes

    Example:
        >>> import torch
        >>> import dietgpu
        >>> dev = torch.device("cuda:0")
        >>> t = torch.randn(10000, dtype=torch.float16, device=dev)
        >>> splits = torch.tensor([3000, 4000, 3000], dtype=torch.int32)
        >>> comp_list, sizes, mem = dietgpu.compress_data_split_size(True, t, splits)
    """
    return torch.ops.dietgpu.compress_data_split_size(
        compress_as_float,
        t_in,
        t_in_split_sizes,
        checksum,
        temp_mem,
        out_compressed,
        out_compressed_bytes,
    )


def compress_data_simple(
    compress_as_float: bool,
    ts_in: list[Tensor],
    checksum: bool = False,
    temp_mem: int = 67108864,
) -> list[Tensor]:
    """
    Simple compression API that returns individual compressed tensors.

    Args:
        compress_as_float: If True, compress as float data
        ts_in: List of input tensors to compress
        checksum: Whether to include checksum in compressed data
        temp_mem: Optional temporary memory size in bytes (default: 64MB)

    Returns:
        List of 1D uint8 tensors, one per input tensor

    Example:
        >>> import torch
        >>> import dietgpu
        >>> dev = torch.device("cuda:0")
        >>> tensors = [torch.randn(1000, dtype=torch.bfloat16, device=dev)]
        >>> compressed = dietgpu.compress_data_simple(True, tensors)
    """
    return torch.ops.dietgpu.compress_data_simple(
        compress_as_float, ts_in, checksum, temp_mem
    )


# ============================================================================
# Decompression Functions
# ============================================================================


def decompress_data(
    compress_as_float: bool,
    ts_in: list[Tensor],
    ts_out: list[Tensor],
    checksum: bool = False,
    temp_mem: Tensor | None = None,
    out_status: Tensor | None = None,
    out_decompressed_words: Tensor | None = None,
) -> int:
    """
    Decompress a batch of compressed tensors.

    Args:
        compress_as_float: If True, decompress as float data
        ts_in: List of compressed uint8 tensors
        ts_out: List of output tensors (pre-allocated with correct dtype and size)
        checksum: Whether to verify checksum
        temp_mem: Optional temporary memory buffer (uint8 tensor on CUDA)
        out_status: Optional 1D uint8 tensor to store decompression status (1 byte per tensor)
        out_decompressed_words: Optional 1D int32 tensor to store decompressed sizes

    Returns:
        Amount of temporary memory used in bytes

    Example:
        >>> import torch
        >>> import dietgpu
        >>> dev = torch.device("cuda:0")
        >>> # Assuming comp is compressed data from compress_data
        >>> out_tensors = [torch.empty(1000, dtype=torch.float16, device=dev)]
        >>> mem_usage = dietgpu.decompress_data(True, comp, out_tensors, checksum=True)
    """
    return torch.ops.dietgpu.decompress_data(
        compress_as_float,
        ts_in,
        ts_out,
        checksum,
        temp_mem,
        out_status,
        out_decompressed_words,
    )


def decompress_data_split_size(
    compress_as_float: bool,
    ts_in: list[Tensor],
    t_out: Tensor,
    t_out_split_sizes: Tensor,
    checksum: bool = False,
    temp_mem: Tensor | None = None,
    out_status: Tensor | None = None,
    out_decompressed_words: Tensor | None = None,
) -> int:
    """
    Decompress multiple compressed tensors into a single output tensor.

    Args:
        compress_as_float: If True, decompress as float data
        ts_in: List of compressed uint8 tensors
        t_out: Output tensor (pre-allocated with correct dtype and total size)
        t_out_split_sizes: 1D int32 tensor on CPU with original split sizes
        checksum: Whether to verify checksum
        temp_mem: Optional temporary memory buffer (uint8 tensor on CUDA)
        out_status: Optional 1D uint8 tensor to store decompression status
        out_decompressed_words: Optional 1D int32 tensor to store decompressed sizes

    Returns:
        Amount of temporary memory used in bytes

    Example:
        >>> import torch
        >>> import dietgpu
        >>> dev = torch.device("cuda:0")
        >>> # Assuming comp_list is from compress_data_split_size
        >>> out = torch.empty(10000, dtype=torch.float16, device=dev)
        >>> splits = torch.tensor([3000, 4000, 3000], dtype=torch.int32)
        >>> mem = dietgpu.decompress_data_split_size(True, comp_list, out, splits)
    """
    return torch.ops.dietgpu.decompress_data_split_size(
        compress_as_float,
        ts_in,
        t_out,
        t_out_split_sizes,
        checksum,
        temp_mem,
        out_status,
        out_decompressed_words,
    )


def decompress_data_simple(
    compress_as_float: bool,
    ts_in: list[Tensor],
    checksum: bool = False,
    temp_mem: int = 67108864,
) -> list[Tensor]:
    """
    Simple decompression API that automatically allocates output tensors.

    Args:
        compress_as_float: If True, decompress as float data
        ts_in: List of compressed uint8 tensors
        checksum: Whether to verify checksum
        temp_mem: Optional temporary memory size in bytes (default: 64MB)

    Returns:
        List of decompressed tensors with appropriate dtypes and sizes

    Example:
        >>> import torch
        >>> import dietgpu
        >>> # Assuming comp is compressed data
        >>> decompressed = dietgpu.decompress_data_simple(True, comp)
    """
    return torch.ops.dietgpu.decompress_data_simple(
        compress_as_float, ts_in, checksum, temp_mem
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Compression functions
    "compress_data",
    "compress_data_simple",
    "compress_data_split_size",
    # Decompression functions
    "decompress_data",
    "decompress_data_simple",
    "decompress_data_split_size",
    "max_any_compressed_output_size",
    "max_any_compressed_size",
    # Utility functions
    "max_float_compressed_output_size",
    "max_float_compressed_size",
]
