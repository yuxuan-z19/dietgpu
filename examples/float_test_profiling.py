# Copyright (c) 2026 Yee Hin Chong (yuxuan-z)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import statistics
import unittest

import torch

import dietgpu


def profile_compression_decompression(
    dev: torch.device,
    ts: list[torch.Tensor],
    temp_mem: torch.Tensor | None = None,
    test_name: str = "",
    warmup_iters: int = 16,
    profile_iters: int = 64,
):
    """
    Profile compression and decompression with detailed metrics using CUDA events
    """
    print(f"\n{'=' * 80}")
    print(f"Test: {test_name}")
    print(f"{'=' * 80}")
    print(f"Warmup iterations: {warmup_iters}, Profile iterations: {profile_iters}")

    compress_times = []
    decompress_times = []
    compress_peak_memories = []
    decompress_peak_memories = []
    compression_ratios = []

    for idx, t in enumerate(ts):
        t_bytes = t.numel() * t.element_size()

        # ===== Warmup Phase =====
        for _ in range(warmup_iters):
            if temp_mem is not None:
                comp, sizes, _ = dietgpu.compress_data(True, [t], True, temp_mem)
            else:
                comp, sizes, _ = dietgpu.compress_data(True, [t], True, None)

            out_t = torch.empty(t.size(), dtype=t.dtype, device=t.device)
            compressed_t = comp[0]
            compressed_size = sizes[0].item()
            num_elements = compressed_size // compressed_t.element_size()
            truncated_comp = compressed_t.narrow(0, 0, num_elements).clone()

            if temp_mem is not None:
                out_status = torch.empty([1], dtype=torch.uint8, device=dev)
                out_sizes = torch.empty([1], dtype=torch.int32, device=dev)
                dietgpu.decompress_data(
                    True,
                    [truncated_comp],
                    [out_t],
                    True,
                    temp_mem,
                    out_status,
                    out_sizes,
                )
            else:
                dietgpu.decompress_data(True, [truncated_comp], [out_t], True)

        torch.cuda.synchronize()

        # ===== Profiling Phase =====
        for _ in range(profile_iters):
            # Measure compression
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            if temp_mem is not None:
                comp, sizes, _ = dietgpu.compress_data(True, [t], True, temp_mem)
            else:
                comp, sizes, _ = dietgpu.compress_data(True, [t], True, None)
            end_event.record()

            torch.cuda.synchronize()
            compress_time = start_event.elapsed_time(end_event)  # milliseconds

            compressed_size = sizes[0].item()
            compression_ratio = t_bytes / compressed_size

            # Prepare for decompression
            out_t = torch.empty(t.size(), dtype=t.dtype, device=t.device)
            compressed_t = comp[0]
            num_elements = compressed_size // compressed_t.element_size()
            truncated_comp = compressed_t.narrow(0, 0, num_elements).clone()

            # Measure decompression
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            if temp_mem is not None:
                out_status = torch.empty([1], dtype=torch.uint8, device=dev)
                out_sizes = torch.empty([1], dtype=torch.int32, device=dev)
                dietgpu.decompress_data(
                    True,
                    [truncated_comp],
                    [out_t],
                    True,
                    temp_mem,
                    out_status,
                    out_sizes,
                )
            else:
                dietgpu.decompress_data(True, [truncated_comp], [out_t], True)
            end_event.record()

            torch.cuda.synchronize()
            decompress_time = start_event.elapsed_time(end_event)  # milliseconds

            # Get peak memory
            compress_peak_memory = torch.cuda.max_memory_allocated(dev)
            decompress_peak_memory = torch.cuda.max_memory_allocated(dev)

            # Verify correctness
            assert torch.equal(t, out_t), f"Data mismatch at index {idx}"

            # Store results
            compress_times.append(compress_time)
            decompress_times.append(decompress_time)
            compress_peak_memories.append(compress_peak_memory / (1024 * 1024))
            decompress_peak_memories.append(decompress_peak_memory / (1024 * 1024))
            compression_ratios.append(compression_ratio)

        # Calculate statistics
        def calc_stats(data):
            return {
                "mean": statistics.mean(data),
                "median": statistics.median(data),
                "std": statistics.stdev(data) if len(data) > 1 else 0,
                "min": min(data),
                "max": max(data),
            }

        compress_time_stats = calc_stats(compress_times)
        decompress_time_stats = calc_stats(decompress_times)
        compress_mem_stats = calc_stats(compress_peak_memories)
        decompress_mem_stats = calc_stats(decompress_peak_memories)
        compression_ratio_avg = statistics.mean(compression_ratios)

        # Calculate throughput
        compress_throughput = (t_bytes / (1024**3)) / (
            compress_time_stats["mean"] / 1000
        )
        decompress_throughput = (t_bytes / (1024**3)) / (
            decompress_time_stats["mean"] / 1000
        )

        # Print results
        print(f"\nTensor {idx + 1} ({t_bytes / (1024**2):.2f} MB):")
        print(f"  Compression ratio: {compression_ratio_avg:.2f}x")
        print("\n  Compression Time (ms):")
        print(
            f"    Mean: {compress_time_stats['mean']:.3f} | "
            f"Median: {compress_time_stats['median']:.3f} | "
            f"Std: {compress_time_stats['std']:.3f} | "
            f"Min: {compress_time_stats['min']:.3f} | "
            f"Max: {compress_time_stats['max']:.3f}"
        )
        print(f"  Compression Throughput: {compress_throughput:.2f} GB/s")
        print("\n  Compression Peak Memory (MB):")
        print(
            f"    Mean: {compress_mem_stats['mean']:.2f} | "
            f"Median: {compress_mem_stats['median']:.2f} | "
            f"Std: {compress_mem_stats['std']:.2f} | "
            f"Min: {compress_mem_stats['min']:.2f} | "
            f"Max: {compress_mem_stats['max']:.2f}"
        )
        print("\n  Decompression Time (ms):")
        print(
            f"    Mean: {decompress_time_stats['mean']:.3f} | "
            f"Median: {decompress_time_stats['median']:.3f} | "
            f"Std: {decompress_time_stats['std']:.3f} | "
            f"Min: {decompress_time_stats['min']:.3f} | "
            f"Max: {decompress_time_stats['max']:.3f}"
        )
        print(f"  Decompression Throughput: {decompress_throughput:.2f} GB/s")
        print("\n  Decompression Peak Memory (MB):")
        print(
            f"    Mean: {decompress_mem_stats['mean']:.2f} | "
            f"Median: {decompress_mem_stats['median']:.2f} | "
            f"Std: {decompress_mem_stats['std']:.2f} | "
            f"Min: {decompress_mem_stats['min']:.2f} | "
            f"Max: {decompress_mem_stats['max']:.2f}"
        )

    print(f"\n{'=' * 80}\n")

    return {
        "compress_times": compress_times,
        "decompress_times": decompress_times,
        "compression_ratios": compression_ratios,
    }


class TestFloatCodecProfiling(unittest.TestCase):
    def test_codec_profiling(self):
        """Test compression/decompression with profiling for different dtypes"""
        dev = torch.device("cuda:0")
        temp_mem = torch.empty([64 * 1024 * 1024], dtype=torch.uint8, device=dev)

        for dt in [torch.bfloat16, torch.float16, torch.float32]:
            print(f"\n{'#' * 80}")
            print(f"# Testing dtype: {dt}")
            print(f"{'#' * 80}")

            for use_temp_mem in [False, True]:
                mode_name = (
                    "with_temp_memory" if use_temp_mem else "without_temp_memory"
                )
                ts = [
                    torch.normal(0, 1.0, [i], dtype=dt, device=dev)
                    for i in [10000, 100000, 1000000]
                ]

                test_name = f"{dt}_{mode_name}"
                temp_mem_arg = temp_mem if use_temp_mem else None

                profile_compression_decompression(
                    dev, ts, temp_mem_arg, test_name, warmup_iters=16, profile_iters=64
                )

    def test_large_tensor_profiling(self):
        """Test with large tensors"""
        dev = torch.device("cuda:0")
        temp_mem = torch.empty([64 * 1024 * 1024], dtype=torch.uint8, device=dev)

        for dt in [torch.bfloat16, torch.float16, torch.float32]:
            print(f"\n{'#' * 80}")
            print(f"# Large tensor test - dtype: {dt}")
            print(f"{'#' * 80}")

            for use_temp_mem in [False, True]:
                mode_name = (
                    "with_temp_memory" if use_temp_mem else "without_temp_memory"
                )
                ts = [torch.normal(0, 1.0, [10000000], dtype=dt, device=dev)]

                test_name = f"large_{dt}_{mode_name}"
                temp_mem_arg = temp_mem if use_temp_mem else None

                profile_compression_decompression(
                    dev, ts, temp_mem_arg, test_name, warmup_iters=16, profile_iters=64
                )

    def test_multi_size_serial_compression(self):
        """Test multi-size serial compression (single tensor mode)"""
        dev = torch.device("cuda:0")
        temp_mem = torch.empty([256 * 1024 * 1024], dtype=torch.uint8, device=dev)

        # Data sizes in MB
        sizes_mb = [128, 256, 512, 1024, 2048]

        print(f"\n{'#' * 80}")
        print("# Multi-size Serial Compression Test (Single Tensor Mode)")
        print(f"{'#' * 80}")

        for dt in [torch.bfloat16, torch.float16, torch.float32]:
            print(f"\n{'=' * 80}")
            print(f"Testing dtype: {dt}")
            print(f"{'=' * 80}")

            for use_temp_mem in [False, True]:
                mode_name = (
                    "with_temp_memory" if use_temp_mem else "without_temp_memory"
                )
                temp_mem_arg = temp_mem if use_temp_mem else None

                print(f"\n--- Mode: {mode_name} ---")

                for size_mb in sizes_mb:
                    # Calculate number of elements based on dtype
                    bytes_per_element = dt.itemsize
                    num_elements = (size_mb * 1024 * 1024) // bytes_per_element

                    # Create single tensor
                    t = torch.normal(0, 1.0, [num_elements], dtype=dt, device=dev)

                    test_name = f"serial_{dt}_{mode_name}_{size_mb}MB"
                    profile_compression_decompression(
                        dev,
                        [t],
                        temp_mem_arg,
                        test_name,
                        warmup_iters=16,
                        profile_iters=64,
                    )

    def test_multi_size_batch_compression(self):
        """Test multi-size batch compression (batch mode)"""
        dev = torch.device("cuda:0")
        temp_mem = torch.empty([256 * 1024 * 1024], dtype=torch.uint8, device=dev)

        # Data sizes in MB
        sizes_mb = [128, 256, 512, 1024, 2048]
        batch_size = 10

        print(f"\n{'#' * 80}")
        print(
            f"# Multi-size Batch Compression Test (Batch Mode, batch_size={batch_size})"
        )
        print(f"{'#' * 80}")

        for dt in [torch.bfloat16, torch.float16, torch.float32]:
            print(f"\n{'=' * 80}")
            print(f"Testing dtype: {dt}")
            print(f"{'=' * 80}")

            for use_temp_mem in [False, True]:
                mode_name = (
                    "with_temp_memory" if use_temp_mem else "without_temp_memory"
                )
                temp_mem_arg = temp_mem if use_temp_mem else None

                print(f"\n--- Mode: {mode_name} ---")

                for size_mb in sizes_mb:
                    # Calculate number of elements per tensor
                    bytes_per_element = dt.itemsize
                    num_elements = (size_mb * 1024 * 1024) // bytes_per_element

                    # Create batch of tensors
                    ts = [
                        torch.normal(0, 1.0, [num_elements], dtype=dt, device=dev)
                        for _ in range(batch_size)
                    ]

                    test_name = f"batch_{dt}_{mode_name}_{size_mb}MB"
                    profile_compression_decompression(
                        dev,
                        ts,
                        temp_mem_arg,
                        test_name,
                        warmup_iters=16,
                        profile_iters=64,
                    )

    def test_pipeline_compression_mode(self):
        """Test pipeline mode: sequential compression/decompression of multiple tensors"""
        dev = torch.device("cuda:0")
        temp_mem = torch.empty([256 * 1024 * 1024], dtype=torch.uint8, device=dev)

        # Data sizes in MB
        sizes_mb = [128, 256, 512, 1024]
        num_tensors = 5

        print(f"\n{'#' * 80}")
        print(f"# Pipeline Compression Mode Test (Sequential {num_tensors} tensors)")
        print(f"{'#' * 80}")

        for dt in [torch.bfloat16, torch.float16, torch.float32]:
            print(f"\n{'=' * 80}")
            print(f"Testing dtype: {dt}")
            print(f"{'=' * 80}")

            for use_temp_mem in [False, True]:
                mode_name = (
                    "with_temp_memory" if use_temp_mem else "without_temp_memory"
                )
                temp_mem_arg = temp_mem if use_temp_mem else None

                print(f"\n--- Mode: {mode_name} ---")

                for size_mb in sizes_mb:
                    # Calculate number of elements
                    bytes_per_element = dt.itemsize
                    num_elements = (size_mb * 1024 * 1024) // bytes_per_element

                    # Create multiple tensors
                    tensors = [
                        torch.normal(0, 1.0, [num_elements], dtype=dt, device=dev)
                        for _ in range(num_tensors)
                    ]

                    test_name = f"pipeline_{dt}_{mode_name}_{size_mb}MB"
                    profile_compression_decompression(
                        dev,
                        tensors,
                        temp_mem_arg,
                        test_name,
                        warmup_iters=16,
                        profile_iters=64,
                    )


if __name__ == "__main__":
    unittest.main()
