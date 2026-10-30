# DietGPU: GPU-based lossless compression for numerical data

**Original Author:** Jeff Johnson (@wickedfoo), `jhj _at_ fb.com`  
**Fork Maintainer:** Yee Hin Chong (@yuxuan-z19), `yuxuan-z19 _at_ qq.com`

> **Note:** This is a fork of the original [DietGPU](https://github.com/facebookresearch/dietgpu) project by Meta Platforms, which has been archived. This fork includes modernization updates, improved packaging, and additional features, including direct support for CUDA-compatible acclerators (e.g. T-Head SAIL series).

DietGPU is a library for fast specialized lossless compression of data on CUDA-compatible devices, meant for ML/HPC applications. It also contains the first publicly available GPU-based generalized [asymmetric numeral system (ANS)](https://en.wikipedia.org/wiki/Asymmetric_numeral_systems) compressor and decompressor. It is a GPU analogue to Yann Collet's [FSE (Finite State Entropy)](https://github.com/Cyan4973/FiniteStateEntropy) ANS library.

It currently consists of two parts:

- **ANS entropy codec**: a generalized byte-oriented range-based ANS (rANS) entropy encoder and decoder, that operates at throughputs around 250-410 GB/s for reasonable data sizes on an A100 GPU.
- **Floating point codec**: an extension to the above to handle fast lossless compression and decompression of unstructured floating point data, for use in ML and HPC applications, especially in communicating over local interconnects (PCIe / NVLink) and remote interconnects (Ethernet / InfiniBand). This operates at around 250-600 GB/s for reasonable data sizes on an A100 GPU.

Both APIs are available in both C++ (raw device pointers) and Python/PyTorch (PyTorch tensor) API forms.

## Quick Start

### Basic Compression

```python
import torch
import dietgpu

dev = torch.device("cuda:0")

# Create test data
tensors = [
    torch.randn(10000, dtype=torch.float16, device=dev),
    torch.randn(20000, dtype=torch.float16, device=dev),
]

# Simple compression
compressed = dietgpu.compress_data_simple(True, tensors, checksum=True)
print(f"Compressed {len(tensors)} tensors")

# Simple decompression
decompressed = dietgpu.decompress_data_simple(True, compressed, checksum=True)

# Verify
for orig, decomp in zip(tensors, decompressed):
    assert torch.equal(orig, decomp)
print("✓ Data verified!")
```

### Batch Compression with Full Control

```python
import torch
import dietgpu

dev = torch.device("cuda:0")
tensors = [torch.randn(10000, dtype=torch.bfloat16, device=dev) for _ in range(5)]

# Allocate temporary memory for better performance
temp_mem = torch.empty([64 * 1024 * 1024], dtype=torch.uint8, device=dev)

# Compress
comp, sizes, mem_usage = dietgpu.compress_data(
    compress_as_float=True,
    ts_in=tensors,
    checksum=True,
    temp_mem=temp_mem
)

print(f"Compressed {len(tensors)} tensors")
print(f"Memory used: {mem_usage / 1024 / 1024:.2f} MB")

# Prepare output tensors
out_tensors = [torch.empty(t.size(), dtype=t.dtype, device=dev) for t in tensors]

# Decompress
dietgpu.decompress_data(
    compress_as_float=True,
    ts_in=[comp],
    ts_out=out_tensors,
    checksum=True,
    temp_mem=temp_mem
)
```

### Split Compression

```python
import torch
import dietgpu

dev = torch.device("cuda:0")

# Create a large tensor
data = torch.randn(100000, dtype=torch.float32, device=dev)

# Define split sizes
split_sizes = torch.tensor([30000, 40000, 30000], dtype=torch.int32)

# Compress as splits
comp_list, sizes, mem = dietgpu.compress_data_split_size(
    compress_as_float=True,
    t_in=data,
    t_in_split_sizes=split_sizes,
    checksum=True
)

print(f"Split into {len(comp_list)} parts")

# Decompress back
output = torch.empty_like(data)
dietgpu.decompress_data_split_size(
    compress_as_float=True,
    ts_in=comp_list,
    t_out=output,
    t_out_split_sizes=split_sizes,
    checksum=True
)

assert torch.equal(data, output)
print("✓ Split compression verified!")
```

## Supported Data Types

- `torch.float16` (half precision)
- `torch.bfloat16` (brain float16)
- `torch.float32` (single precision)
- Arbitrary byte data (when `compress_as_float=False`)

## Performance Tips

1. **Use temporary memory**: Pre-allocating `temp_mem` can improve performance by avoiding repeated allocations
2. **Batch operations**: Compress multiple tensors together for better GPU utilization
3. **Checksum**: Enable checksum for data integrity verification (small performance overhead)
4. **Memory alignment**: For byte-level compression, ensure 16-byte alignment for best performance

## Documentation

Documentation is [available in the wiki](https://github.com/facebookresearch/dietgpu/wiki).

See the [examples/](examples/) directory for more usage examples.

## Installation

> **Why `--no-build-isolation`?** DietGPU links `libtorch` directly, so it
> must be compiled against the *same* PyTorch that will later import it.
> PEP 517 build isolation hides your real environment; if `torch` were a
> build requirement it would pull the newest version instead, producing a
> package that installs fine but dies on import:
>
> ```
> OSError: libdietgpu.so: undefined symbol: _ZN3c104impl22getGlobalPyInterpreterEv
> ```
>
> The sections below show how to install with isolation off for each
> scenario. Within the project tree, `[tool.uv]` in this repo's
> `pyproject.toml` handles it automatically; everywhere else the flag must
> be passed by hand.

### Quick Install from PyPI or a Local Tarball

dietgpu is distributed as an **sdist only** (no pre-built wheels). Whether the
source comes from PyPI or a downloaded `dietgpu-*.tar.gz` (GitHub Releases, a
private index, or `python -m build --sdist`), pip/uv will compile it locally.

Both cases need the same `--no-build-isolation` flag, for the same reason:
neither a PyPI sdist nor a tarball carries a workspace `pyproject.toml` that
uv can read, so the `[tool.uv] no-build-isolation-package` setting in this
repo has no effect. Build isolation stays on unless you turn it off
explicitly -- and isolation would hide your real PyTorch from the build (see
the callout above for what goes wrong).

> **Tip:** If your own project depends on dietgpu, add this once to your
> `pyproject.toml` so you never need the flag by hand:
>
> ```toml
> [tool.uv]
> no-build-isolation-package = ["dietgpu"]
> ```
>
> With that in place, any `uv pip install` that resolves dietgpu -- from
> PyPI, a tarball, or a local path -- builds it without isolation
> automatically.

#### PyPI

```bash
# uv still fetches scikit-build-core/cmake/ninja for you -- they are runtime
# dependencies of the sdist.
uv pip install dietgpu --no-build-isolation

# pip: install the build tools once, then disable isolation
pip install scikit-build-core ninja cmake
pip install dietgpu --no-build-isolation
```

#### Local tarball

```bash
uv pip install --no-build-isolation /path/to/dietgpu-0.1.0.tar.gz
pip install --no-build-isolation /path/to/dietgpu-0.1.0.tar.gz
```

Alternatively, extract first and let uv pick up the `[tool.uv]` setting
from the unpacked tree:

```bash
tar xzf dietgpu-0.1.0.tar.gz && cd dietgpu-0.1.0
uv pip install .   # [tool.uv] in pyproject.toml applies here
```

### Install from Source (Development Mode)

```bash
# Clone the repository
git clone --recursive https://github.com/yuxuan-z19/dietgpu.git
cd dietgpu

# One command, no flags: uv installs torch and the build toolchain into the
# target environment, then compiles against them.
uv pip install -e .

# Or using pip (build tools must already be present)
pip install scikit-build-core ninja cmake
pip install -e . --no-build-isolation
```

The installation will automatically:
- Use Ninja build system (automatically managed by scikit-build-core)
- Build the C++ library using CMake
- Compile CUDA kernels
- Install Python package and shared libraries

**That's it!** No manual CMake or make commands needed.

### Parallel Build Configuration

Compilation is parallel by default: Ninja uses roughly one job per CPU core. To
cap it, use CMake's own variable:

```bash
# Cap the number of parallel compile jobs
export CMAKE_BUILD_PARALLEL_LEVEL=8
uv pip install -e .

# Or set it inline
CMAKE_BUILD_PARALLEL_LEVEL=16 uv pip install -e .
```

> **`MAX_JOBS` does nothing here.** It is the convention from
> `torch.utils.cpp_extension`, but scikit-build-core never reads it, and
> `CMAKE_BUILD_PARALLEL_LEVEL` is an *environment* variable — setting it inside
> CMake has no effect on `cmake --build`. Use the env var above.

### Skipping the C++ Tests

The gtest executables are built by default, which also builds googletest itself
(four shared libraries). Turn them off to skip that:

```bash
CMAKE_ARGS="-DDIETGPU_BUILD_TESTS=OFF" uv pip install -e .
```

### Selecting the GPU Architecture

By default the build probes the GPUs installed in the machine running the
compiler and targets exactly those. That is the wrong question whenever the
build machine is not the machine that will run dietgpu — CI runners, containers
without device access, cross compilation — and on CUDA-compatible non-NVIDIA
accelerators such as the Alibaba T-Head PPU. Pin the target explicitly instead:

```bash
# PyTorch's standard dotted form. Recommended: this one knob drives both
# PyTorch's flags and dietgpu's own, so you get a single-architecture build.
TORCH_CUDA_ARCH_LIST="8.9" uv pip install dietgpu --no-build-isolation

# CMake's numeric form, when you need its -real / -virtual suffixes.
DIETGPU_CUDA_ARCHITECTURES="89" uv pip install -e .

# Several architectures at once (each one multiplies build time and size).
TORCH_CUDA_ARCH_LIST="7.0;8.0;8.9" uv pip install -e .
DIETGPU_CUDA_ARCHITECTURES="70;80;89" uv pip install -e .

# The pip equivalents need the isolation flag spelled out:
TORCH_CUDA_ARCH_LIST="8.9" pip install dietgpu --no-build-isolation
DIETGPU_CUDA_ARCHITECTURES="89" pip install -e . --no-build-isolation
```

For a manual CMake build, pass the same two as `-D` options:
`cmake .. -DTORCH_CUDA_ARCH_LIST=8.9` or `cmake .. -DDIETGPU_CUDA_ARCHITECTURES=89`.

Notes:

- Prefer `TORCH_CUDA_ARCH_LIST`. Setting only `DIETGPU_CUDA_ARCHITECTURES`
  leaves PyTorch free to append an architecture of its own (probed from the
  build machine), so the result contains both.
- With no GPU visible *and* neither variable set, CMake falls back to **every**
  architecture that `nvcc` supports. dietgpu warns about this and the build
  still succeeds, but it is many times slower and the binaries much larger —
  pin one of the two variables.
- The probe reports one entry per device, so a homogeneous multi-GPU machine
  (e.g. 8× RTX 4090) is de-duplicated to a single architecture automatically.

### System Requirements

- **Python** >= 3.11
- **PyTorch** >= 2.0.0 (with CUDA support)
- **CUDA Toolkit** >= 11.0
- **CMake** >= 3.22 (CMake 3.22 is required by third_party/ng-log (ng-log v0.8.4);
  dietgpu uses `DIETGPU_CUDA_ARCHITECTURES` instead of `CMAKE_CUDA_ARCHITECTURES`
  to dodge CMP0104 pre-population, but the 3.18 floor is still required for the
  underlying CMake feature to exist. No upper bound: ng-log uses
  `cmake_policy(VERSION 3.16...3.27)`, compatible with CMake 4.x). Installed
  for you by `uv pip install`; only the pip route
  needs it present beforehand.
- **Ninja** (or GNU Make as a fallback). Also installed for you by uv.
- **C++ Compiler** (with C++17 support)
- **CUDA-compatible GPU** (for actual compression/decompression)
  - Tested on: NVIDIA GPUs (A100, RTX4090, etc.)
  - Also compatible with: Alibaba T-Head PPU and other CUDA-compatible accelerators

### Verify Installation

```python
import torch
import dietgpu

print(f"DietGPU version: {dietgpu.__version__}")

# Test compression
dev = torch.device("cuda:0")
tensors = [torch.randn(10000, dtype=torch.float16, device=dev)]
compressed = dietgpu.compress_data_simple(True, tensors)
print(f"✓ Compression successful!")
```

## Building from Source (Manual)

If you prefer to build manually or need to customize the build:

### Manual CMake Build

Clone this repo using

```shell
git clone --recursive https://github.com/facebookresearch/dietgpu
cd dietgpu
mkdir build && cd build
cmake .. -G Ninja
cmake --build . --target all
```

If you get complaints about `TorchConfig.cmake` then your `CMAKE_PREFIX_PATH` doesn't have the right paths; run

```shell
python -c 'import torch;print(torch.utils.cmake_prefix_path)'
```

to discover where `TorchConfig.cmake` lives (and add that path to your `CMAKE_PREFIX_PATH`).
In general, you can run
```shell
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../:$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')"
```

If you get complaints about `/dietgpu/third_party/glog... does not contain a CMakeLists.txt file.` then you didn't pull the submodules; run

```shell
git submodule sync
git submodule update --init --recursive --jobs 0
```
and try again.

## Library rationale

As on-device global memory / HBM bandwidth continues to improve at a faster rate than CPU/GPU interconnect or server-to-server networking bandwidth, spending GPU compute and gmem bandwidth to save on data sent over interconnects is becoming more advantageous. DietGPU aims to target this gap.

One can imagine a Pareto-optimal tradeoff curve between realizable compression ratios versus speed. On one end of the curve exists algorithms for supporting arbitrary data using dictionary/LZ type compression like some of the techniques in [Nvidia's nvCOMP](https://github.com/NVIDIA/nvcomp) at potentially high compression rates. At another end of the curve, one can imagine use completely on-device as something like a 1990s-style virtual RAM extender, where achievable compression is only 0.6x-0.9x or so, but compression can operate at around 1/4x to 1/2x the peak global memory bandwidth of the GPU. We emphasize the latter, where speed rather than compression ratio is important, where we can compress data that is even sent between GPUs in a single server over NVLink or PCIe. The savings may be low, but the effective network speed can be increased by 10-30%. For large-scale neural network training on hundreds of GPUs, this could translate into an additional 5-10% end-to-end performance increase.

The initial focus of this library will be in HPC/ML distributed collective communications libraries, for primitives such as all-to-all, all-gather, reduce-scatter and all-reduce. Right now no off the shelf integration is provided (in progress), but the basics of the C++ API are available for use, as are Python-level PyTorch tensor-based APIs.

## ANS codec

The rANS codec operates on 8 bit bytes. It can compress arbitrary data, but using statistics gathered on a bytewise basis, so data highly structured or redundant at a level above byte level will typically not compress well. This codec however is meant to be applicable for any number of lossless compression applications, including usage as an entropy coder for LZ or RLE type matches for a fully-formed compression system. Symbol probability precisions supported are 9, 10 and 11 bits (i.e., symbol occurances are quantized to the nearest 1/512, 1/1024 or 1/2048).

## Float codec

The floating point compressor at the moment uses the rANS codec to handle compression of floating point exponents, as typically in ML/HPC data a very limited exponent dynamic range is used and is highly compressible. Floating point sign and significand values tend to be less compressible / fairly high entropy in practice, though sparse data or presence of functions like ReLU in neural networks can result in a lot of outright zero values which are very compressible. A future extension to the library will allow for specialized compression of sparse or semi-sparse data, specializing compression of zeros. At the moment only float16 (IEEE 754 binary16) and bfloat16 (fields of the most significant 16 bits of a IEEE 754 binary32 word) are supported, with float32 (IEEE 754 binary32) support coming shortly.

## API design

The basics of the design should work on CC 3.5+ (Kepler class) GPUs or later, though it has been primarily developed for and has only been tested on V100/A100 GPUs.

Both APIs are available in both C++ (raw pointers) and Python/PyTorch (PyTorch tensor) API forms. It is a batch oriented API; both compression and decompression operate in batches of independent arrays of data which are independently compressed or decompressed, though with the floating point compressor, all arrays in the batch must be of the same data type. ANS compression symbol probabilities are calculated independently for each array in the batch, and each produced output compressed tensor in a batch is independently decompressible (and the ANS statistics are tailored to each individual array in the batch). See the wiki for details.

The APIs are oriented around batching, though providing a large batch size of 1 also results in good performance (in fact, bs > 1 has somewhat worse performance than bs = 1 for sufficiently large data sizes at the moment, due to work imbalance issues). Arrays in the batch can be of arbitrary, varying sizes. The library treats all data as unstructured 1 dimensional arrays, so the PyTorch API does not really care about dimensionality. The primitive unit of compression are 4 KiB segments of the input data, which are assigned to individual warps. Typically, it is not worth using DietGPU unless one has at least 512 KiB of data or so due to compression overheads, and poor performance will be seen unless the total data size (whether bs = 1 or a large batch) is enough such that (total size in bytes / 4 KiB) is on par with the number of concurrently running warps that will saturate a GPUs SMs.

All computation takes place completely on device. The design of the library pays special attention to avoiding memory allocations/deallocations and spurious device-to-host/host-to-device interactions and synchronizations where possible. Assuming inputs and outputs are properly sized and if enough temporary memory scratch space is provided up front, compression and decompression can run completely asynchronously on the GPU without CPU intervention. However, only the GPU during compression knows the actual final compressed size, and a typical application will need to copy the output size buffer containing the final compressed sizes per compression job in the batch in bytes back to the host for use in relocating compressed data elsewhere (in local memory or over the network), so we know how much data to send or copy. As the final output size cannot be predicted in advance, a function is provided to bound the maximum possible compressed output size (which is in fact larger than the input data size) which can be used to allocate an appropriate region of memory for the output. Realizing actual compression savings for applications other than networking would involve an additional memory allocation and memcpy to a new exactly sized buffer.

## Performance

Performance depends upon many factors, including entropy of the input data (higher entropy = more ANS stack memory operations = lower performance), number of SMs on the device and batch/data sizes. Here are some sample runs using an A100 GPU and the sync/alloc-free API on a batch size of 1 from the python PyTorch API, using `torch.normal(0, 1.0, [size], dtype=dt, ...)` to approximate a typical quasi-Gaussian data distribution as seen in real ML data. The float codec for bfloat16 extracts and compresses just the 8 bit exponent, while for float16 it currently operates on the most significant byte of the float word (containing the sign bit, 5 bits of exponent and 2 bits of significand). Typical ML float data might only have 2.7 bits of entropy in the exponent, so the savings ((8 + 2.7) / 16 ~= 0.67x for bfloat16, (11 + 2.7) / 16 ~= 0.85x for float16) is what is seen in the exponent-only strategy.

![non-batch bfloat16 performance](images/dietgpu_bfloat16_nb.png)
![non-batch float16 performance](images/dietgpu_float16_nb.png)

## Planned extensions

- float32 support, possibly float64 support
- compression options to expect semi-sparse floating point data for higher compression (>10% zero values)
- a fused kernel implementation (likely using CUDA cooperative groups) to support single-kernel compression and decompression minimizing temporary memory usage
- a fused kernel implementation using the above to support persistent NCCL-like all-reduce for collective communications libraries
- CUB-like APIs for fusing warp-oriented ANS compression and decompression into arbitrary user kernels
- int32/int64 compression using a fixed-word size LZ-type window history
- support for embedding table compression with sparse reads/row gathers

## References

Prior GPU-based ANS implementations [to my knowledge](https://encode.su/threads/2078-List-of-Asymmetric-Numeral-Systems-implementations) include:

- [GST: GPU-decodable Supercompressed Textures](https://gamma.cs.unc.edu/GST/) (not a generalized ANS codec; meant as part of a texture compression scheme)
- Weissenberger and Schmidt, [Massively Parallel ANS Decoding on GPUs](https://dl.acm.org/doi/10.1145/3337821.3337888) (a decoder only)

Related GPU entropy coder works include:

- Yamamoto et al., [Huffman Coding with Gap Arrays for GPU Acceleration](https://dl.acm.org/doi/10.1145/3404397.3404429)

Related lossless floating point compression works include:

- Lindstrom and Isenburg, [Fast and Efficient Compression of Floating-Point Data](https://computing.llnl.gov/projects/fpzip) (CPU-based)
- Various GPU works from Martin Burtscher's group at Texas State such as Yang et al., [MPC: A Massively Parallel Compression Algorithm for Scientific Data](https://www.semanticscholar.org/paper/MPC%3A-A-Massively-Parallel-Compression-Algorithm-for-Yang-Mukka/1ab6910c90ad714e29954ccd69d569eb2003eb20)

These works are sometimes oriented at compressing HPC-type data (e.g., 2d/3d/Nd grid data) where there may be local/dimensional correlations that can be exploited.

[nvCOMP](https://github.com/NVIDIA/nvcomp), Nvidia's GPU lossless compression library.

## License

DietGPU is licensed with the MIT license, available in the LICENSE file at the top level.

The original work is Copyright (c) Meta Platforms, Inc. and affiliates.
Modifications in this fork are Copyright (c) 2026 Yee Hin Chong (yuxuan-z).

This is an unofficial fork maintained by an individual contributor. It is not endorsed by, sponsored by, or affiliated with Meta Platforms, Inc.
For the upstream project, see https://github.com/facebookresearch/dietgpu.
