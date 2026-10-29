# DietGPU Examples

This directory contains example scripts and tests demonstrating how to use the dietgpu library.

## Running Examples

**Important**: Since dietgpu is installed in the project's virtual environment, use `.venv/bin/python` or `uv run` to execute the scripts:

```bash
pytest examples/float_test.py
```

### Float Compression Test

```bash
pytest examples/float_test.py -v
```

### ANS Compression Test

```bash
pytest examples/ans_test.py
```

### uint8 Compression Test

```bash
# pytest (functional tests)
pytest examples/uint8_test.py -v -s

# or standalone, which prints a per-check report
.venv/bin/python examples/uint8_test.py
```

The uint8 tests cover the byte-wise ANS codec (`compress_as_float=False`):
- Simple, batched, preallocated-buffer and split-size API round-trips
- Compression ratio vs. byte distribution (zeros / skewed / uniform random)
- Checksum detection of a corrupted archive
- Shape flattening to a byte stream, empty tensors, non-multiple-of-4 sizes
- float8 (e4m3fn / e5m2 / fnuz) payloads, including the NaN verification pitfall
- Input requirements (contiguous CUDA tensors) and 64 MB throughput

### Performance Profiling

```bash
# Run all profiling tests (recommended)
python examples/float_test_profiling.py 2>/dev/null

# Or with pytest
pytest examples/float_test_profiling.py -v -s

# Run specific profiling tests
pytest examples/float_test_profiling.py::TestFloatCodecProfiling::test_multi_size_serial_compression -v -s
pytest examples/float_test_profiling.py::TestFloatCodecProfiling::test_multi_size_batch_compression -v -s
pytest examples/float_test_profiling.py::TestFloatCodecProfiling::test_pipeline_compression_mode -v -s
```

The profiling tests include:
- **Multi-size serial compression**: Tests 128MB to 2048MB data sizes in single tensor mode
- **Multi-size batch compression**: Tests batch mode with batch_size=10
- **Pipeline compression mode**: Tests sequential compression/decompression of multiple tensors
- All tests cover bfloat16, float16, and float32 data types
- Each test measures compression/decompression time, throughput, compression ratio, and peak memory usage

### Benchmark Script

```bash
# Run benchmark (standalone script, not pytest)
.venv/bin/python examples/benchmark.py
```

## Available Examples

- **float_test.py** - Pytest tests for float16, bfloat16, and float32 compression (functional tests)
- **ans_test.py** - Pytest tests for byte-level ANS compression (functional tests)
- **uint8_test.py** - Pytest tests and a standalone report for compressing `torch.uint8`
  tensors with the byte-wise ANS codec: correctness, compression ratio per byte
  distribution, checksum behavior, shape handling, float8 payloads and edge cases
- **float_test_profiling.py** - Comprehensive performance profiling with timing, throughput, compression ratio, and memory metrics
  - Multi-size serial/batch compression tests (128MB - 2048MB)
  - Pipeline compression mode tests
  - Coverage for bfloat16, float16, float32 with/without temp memory
- **benchmark.py** - Standalone benchmarking script for performance evaluation (run directly with Python)

## Installation

See the main [README.md](../README.md) for installation instructions.

## API Documentation

See the main [README.md](../README.md) for complete API documentation and usage examples.
