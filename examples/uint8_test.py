# Copyright (c) 2026 Yee Hin Chong (yuxuan-z)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compress / decompress ``torch.uint8`` tensors with DietGPU and verify the result.

``torch.uint8`` is the natural input type for the byte-wise ANS codec, i.e. every
call below passes ``compress_as_float=False``:

* The codec consumes the tensor as a raw byte stream (``numel * element_size``
  bytes) and is lossless, so a uint8 tensor round-trips byte-exactly.
* ``decompress_data_simple`` always rebuilds a ``torch.uint8`` tensor whose
  ``numel`` equals the original number of *bytes*. uint8 is therefore the only
  dtype whose output can be compared against the input directly; float or int32
  tensors can be fed to this path too, but they come back as flat uint8 bytes that
  must be reinterpreted (``t.view(torch.uint8)``) before comparing.
* ANS is an entropy coder: the achievable ratio is set entirely by the byte
  distribution. Skewed data shrinks a lot, uniformly random data expands by ~3%
  instead of compressing, and every archive carries ~544 B of header, so payloads
  below ~4 KB grow rather than shrink.
* Shape is not preserved: a [8, 16, 32] tensor decompresses to a flat [4096] byte
  tensor and has to be viewed back into its original shape.
* The same path is what float8 (e4m3fn / e5m2 and the fnuz variants) has to use,
  since the float codec only supports float16 / bfloat16 / float32; see
  ``test_uint8_float8_payload`` for the view-in / view-out and NaN pitfalls.

Run under pytest::

    pytest examples/uint8_test.py -v -s

or standalone, which prints a report for every check::

    .venv/bin/python examples/uint8_test.py
"""

import torch

import dietgpu

# False selects the byte-wise ANS codec; True would select the float codec, which
# only accepts float16 / bfloat16 / float32.
COMPRESS_AS_FLOAT = False

# Scratch space handed to the codec so that it never falls back to cudaMalloc
# mid-call (which logs "StackDeviceMemory: attempting to allocate ..." warnings).
TEMP_MEM_BYTES = 64 * 1024 * 1024


def make_temp_mem(dev: torch.device) -> torch.Tensor:
    """Allocate a reusable scratch buffer for the non-simple APIs."""
    return torch.empty([TEMP_MEM_BYTES], dtype=torch.uint8, device=dev)


def compress_one(
    t: torch.Tensor, checksum: bool = True, temp_mem: int | None = TEMP_MEM_BYTES
) -> torch.Tensor:
    """Compress a single tensor into a flat uint8 archive."""
    return dietgpu.compress_data_simple(COMPRESS_AS_FLOAT, [t], checksum, temp_mem)[0]


def decompress_one(
    comp: torch.Tensor, checksum: bool = True, temp_mem: int | None = TEMP_MEM_BYTES
) -> torch.Tensor:
    """Decompress a single archive produced by :func:`compress_one`."""
    return dietgpu.decompress_data_simple(
        COMPRESS_AS_FLOAT, [comp], checksum, temp_mem
    )[0]


def report(name: str, in_bytes: int, comp_bytes: int, exact: bool) -> None:
    """Print one achieved-ratio line; ``exact`` is the byte comparison result."""
    ratio = comp_bytes / in_bytes if in_bytes else float("nan")
    print(
        f"  {name:<34} {in_bytes:>9} B -> {comp_bytes:>9} B "
        f"({ratio:.4f}x) exact={exact}"
    )


def check_one(name: str, t: torch.Tensor, checksum: bool = True) -> float:
    """Round-trip one tensor through the simple API, verify it, and report.

    Returns the achieved compressed / uncompressed size ratio.
    """
    comp = compress_one(t, checksum)
    out = decompress_one(comp, checksum)

    exact = (
        out.dtype == torch.uint8
        and out.numel() == t.numel()
        and torch.equal(t.reshape(-1), out)
    )
    assert exact, f"{name}: round-trip was not byte-exact"

    report(name, t.numel(), comp.numel(), exact)
    return comp.numel() / t.numel() if t.numel() else 0.0


def test_uint8_simple_api():
    """Simplest path: one call in, one call out, outputs allocated for you."""
    dev = torch.device("cuda:0")
    print("\n[simple API] compress_data_simple / decompress_data_simple")

    for checksum in [False, True]:
        # temp_mem=None lets the codec cudaMalloc scratch space on demand; passing
        # a buffer keeps it off the allocator (and quiet).
        for temp_mem in [TEMP_MEM_BYTES, None]:
            for n in [1024, 100000, 1 << 20]:
                t = torch.randint(0, 65, [n], dtype=torch.uint8, device=dev)
                comp = compress_one(t, checksum, temp_mem)
                out = decompress_one(comp, checksum, temp_mem)

                assert out.dtype == torch.uint8
                assert out.numel() == n
                assert torch.equal(t, out)
                report(
                    f"n={n} checksum={int(checksum)} temp_mem={temp_mem is not None}",
                    n,
                    comp.numel(),
                    True,
                )


def test_uint8_batch_api():
    """Batched path with caller-provided outputs plus per-tensor status."""
    dev = torch.device("cuda:0")
    temp_mem = make_temp_mem(dev)
    print("\n[batch API] compress_data / decompress_data")

    ts = [
        torch.randint(0, 65, [n], dtype=torch.uint8, device=dev)
        for n in [1000, 4096, 100000, 1 << 20]
    ]

    comp, sizes, temp_usage = dietgpu.compress_data(
        COMPRESS_AS_FLOAT, ts, True, temp_mem
    )
    print(f"  compressed matrix {tuple(comp.shape)}, temp mem used {temp_usage} B")

    # comp is a padded [batch, max_compressed_size] matrix: only the first
    # sizes[i] bytes of row i are real. Truncating to exactly that also proves the
    # reported sizes are accurate, since decompression must succeed on them.
    truncated = [comp[i].narrow(0, 0, sizes[i].item()).clone() for i in range(len(ts))]

    out_ts = [torch.empty(t.size(), dtype=torch.uint8, device=dev) for t in ts]
    out_status = torch.empty([len(ts)], dtype=torch.uint8, device=dev)
    out_sizes = torch.empty([len(ts)], dtype=torch.int32, device=dev)

    dietgpu.decompress_data(
        COMPRESS_AS_FLOAT,
        truncated,
        out_ts,
        True,
        temp_mem,
        out_status,
        out_sizes,
    )

    for t, c, o, status, size in zip(
        ts, truncated, out_ts, out_status, out_sizes, strict=True
    ):
        assert status.item() == 1, "decompression reported failure"
        # For the byte-wise codec the reported size is in bytes, not elements
        assert size.item() == t.numel() * t.element_size()
        assert torch.equal(t, o)
        report(f"n={t.numel()}", t.numel(), c.numel(), True)


def test_uint8_preallocated_compressed_buffer():
    """Reuse a fixed compressed buffer instead of allocating one per call."""
    dev = torch.device("cuda:0")
    temp_mem = make_temp_mem(dev)
    print("\n[preallocated buffer] max_any_compressed_output_size")

    ts = [
        torch.randint(0, 65, [n], dtype=torch.uint8, device=dev)
        for n in [4096, 65536, 1 << 20]
    ]

    rows, cols = dietgpu.max_any_compressed_output_size(ts)
    comp_buf = torch.empty([rows, cols], dtype=torch.uint8, device=dev)
    sizes_buf = torch.zeros([len(ts)], dtype=torch.int32, device=dev)

    # The worst-case bound is far larger than the input (the header reserves
    # per-block state for kDefaultBlockSize blocks), so trust the returned sizes
    # and never cols when storing archives.
    max_in = max(t.numel() for t in ts)
    print(f"  worst-case buffer {rows} x {cols} B for {max_in} B input")

    comp, sizes, _ = dietgpu.compress_data(
        COMPRESS_AS_FLOAT, ts, True, temp_mem, comp_buf, sizes_buf
    )
    assert comp.data_ptr() == comp_buf.data_ptr(), "output buffer was not reused"
    assert sizes.data_ptr() == sizes_buf.data_ptr()

    out_ts = [torch.empty(t.size(), dtype=torch.uint8, device=dev) for t in ts]
    dietgpu.decompress_data(
        COMPRESS_AS_FLOAT,
        [comp[i].narrow(0, 0, sizes[i].item()) for i in range(len(ts))],
        out_ts,
        True,
        temp_mem,
    )

    for t, o, size in zip(ts, out_ts, sizes, strict=True):
        assert torch.equal(t, o)
        report(f"n={t.numel()}", t.numel(), size.item(), True)


def test_uint8_compression_ratio_by_entropy():
    """The ratio tracks the byte distribution; correctness holds in every case."""
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    n = 1 << 22
    print("\n[entropy] compression ratio vs. byte distribution")

    cases = {
        "all zeros": torch.zeros([n], dtype=torch.uint8, device=dev),
        "randint(0, 4)": torch.randint(0, 4, [n], dtype=torch.uint8, device=dev),
        "randint(0, 65)": torch.randint(0, 65, [n], dtype=torch.uint8, device=dev),
        "randint(0, 256)": torch.randint(0, 256, [n], dtype=torch.uint8, device=dev),
        "arange % 256": (torch.arange(n, device=dev) % 256).to(torch.uint8),
    }

    ratios = {name: check_one(name, t) for name, t in cases.items()}

    # Low entropy data compresses well; incompressible data must still round-trip
    assert ratios["all zeros"] < 0.05
    assert ratios["randint(0, 4)"] < 0.4
    assert ratios["randint(0, 65)"] < 0.9
    assert ratios["randint(0, 256)"] >= 1.0, "uniform random data cannot shrink"


def test_uint8_checksum_detects_corruption():
    """checksum=True turns a damaged archive into an error instead of garbage."""
    dev = torch.device("cuda:0")
    t = torch.randint(0, 65, [1 << 20], dtype=torch.uint8, device=dev)
    print("\n[checksum] corruption detection")

    for checksum in [True, False]:
        damaged = compress_one(t, checksum).clone()
        damaged[damaged.numel() // 2] ^= 0xFF

        raised = False
        try:
            out = decompress_one(damaged, checksum)
            # Without a checksum the decode succeeds but the data is silently wrong
            assert not torch.equal(t, out)
        except RuntimeError as e:
            raised = True
            assert "checksum mismatch" in str(e)

        assert raised == checksum
        print(f"  checksum={int(checksum)}: corrupted archive raised={raised}")


def test_uint8_shape_is_flattened_to_bytes():
    """Multi-dimensional inputs come back as a flat uint8 byte stream."""
    dev = torch.device("cuda:0")
    print("\n[shape] multi-dimensional tensors")

    for shape in [[1024], [32, 128], [8, 16, 32]]:
        t = torch.randint(0, 65, shape, dtype=torch.uint8, device=dev)
        comp = compress_one(t)
        out = decompress_one(comp)

        # Only the byte count survives; restore the shape with a view
        assert out.shape == torch.Size([t.numel()])
        assert torch.equal(t, out.view(t.shape))
        report(str(tuple(shape)), t.numel(), comp.numel(), True)
        print(f"    {tuple(t.shape)} -> {tuple(out.shape)} -> view {tuple(t.shape)}")


def test_uint8_float8_payload():
    """float8 tensors use this same byte path; verify them on the uint8 view.

    The float codec only knows float16 / bfloat16 / float32. Its gain comes from
    splitting each multi-byte word into a redundant high byte (ANS coded) plus raw
    low mantissa bytes, and a 1-byte word has nothing to split, so float8 has to go
    through ``compress_as_float=False``. That path accepts a float8 tensor as-is and
    consumes it as raw bytes; viewing to uint8 first gives a byte-identical archive.

    Two things to watch:
    * the archive decodes to a flat uint8 tensor, so restore the dtype with
      ``out.view(fp8_dtype)`` and the shape with ``out.view(shape)``;
    * verify with a byte comparison, never ``torch.equal`` on the float8 view: NaN
      compares unequal to itself, so a byte-exact round-trip of a payload holding
      NaN would be reported as a failure.
    """
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    print("\n[float8] float8 payloads through the byte path")

    fp8_dtypes = [
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    ]

    src = torch.normal(0, 1.0, [1 << 20], device=dev)
    for dt in fp8_dtypes:
        t = src.to(dt)

        # The float codec rejects float8 outright
        try:
            dietgpu.compress_data_simple(True, [t], True, TEMP_MEM_BYTES)
            raise AssertionError(f"{dt}: float codec should have rejected float8")
        except RuntimeError:
            pass

        comp = compress_one(t)
        # Feeding the float8 tensor directly equals feeding its uint8 view
        assert torch.equal(comp, compress_one(t.view(torch.uint8)))

        out = decompress_one(comp)
        assert out.dtype == torch.uint8
        assert torch.equal(t.view(torch.uint8), out)
        assert torch.equal(t, out.view(dt))
        report(str(dt), t.numel(), comp.numel(), True)

    # NaN payloads still round-trip byte-exactly, but the float8 view does not
    # compare equal -- which is exactly why verification belongs on the byte view.
    for dt in [torch.float8_e4m3fn, torch.float8_e5m2]:
        payload = src[: 1 << 16].to(dt).view(torch.uint8).clone()
        payload[0] = 0x7F  # NaN in both e4m3fn and e5m2
        payload[1] = 0xFF
        t = payload.view(dt)
        assert torch.isnan(t.float()).any()

        out = decompress_one(compress_one(t))
        byte_equal = torch.equal(payload, out)
        value_equal = torch.equal(t, out.view(dt))
        assert byte_equal, "float8 payload did not round-trip byte-exactly"
        assert not value_equal, "NaN should have made the float8 view compare unequal"
        print(
            f"  {dt!s} with NaN: byte view equal={byte_equal}, "
            f"float8 view equal={value_equal}"
        )


def test_uint8_split_size_api():
    """Split one big buffer into per-tensor archives without copying it apart."""
    dev = torch.device("cuda:0")
    temp_mem = make_temp_mem(dev)
    print("\n[split] compress_data_split_size / decompress_data_split_size")

    # Interior splits must start on a kANSRequiredAlignment (4 byte) boundary, so
    # every size but the last one has to be a multiple of 4.
    split_sizes = [4096, 8192, 65536, 1000]
    total = sum(split_sizes)
    t = torch.randint(0, 65, [total], dtype=torch.uint8, device=dev)
    sizes_t = torch.IntTensor(split_sizes)

    comp, _, _ = dietgpu.compress_data_split_size(
        COMPRESS_AS_FLOAT, t, sizes_t, True, temp_mem
    )

    out = torch.empty([total], dtype=torch.uint8, device=dev)
    out_status = torch.empty([len(split_sizes)], dtype=torch.uint8, device=dev)
    out_sizes = torch.empty([len(split_sizes)], dtype=torch.int32, device=dev)
    dietgpu.decompress_data_split_size(
        COMPRESS_AS_FLOAT, comp, out, sizes_t, True, temp_mem, out_status, out_sizes
    )

    assert torch.equal(t, out)
    assert out_status.tolist() == [1] * len(split_sizes)
    assert out_sizes.tolist() == split_sizes

    for orig, decomp, c in zip(
        torch.split(t, split_sizes),
        torch.split(out, split_sizes),
        comp,
        strict=True,
    ):
        assert torch.equal(orig, decomp)
        report(f"split n={orig.numel()}", orig.numel(), c.numel(), True)

    # An unaligned interior split is rejected up front
    try:
        dietgpu.compress_data_split_size(
            COMPRESS_AS_FLOAT,
            t,
            torch.IntTensor([4095, total - 4095]),
            True,
            temp_mem,
        )
        raise AssertionError("unaligned interior split should have been rejected")
    except RuntimeError as e:
        print(f"  unaligned interior split rejected: {str(e).splitlines()[0]}")


def test_uint8_edge_cases():
    """Empty tensors, sizes that are not a multiple of 4, and input requirements."""
    dev = torch.device("cuda:0")
    print("\n[edge cases]")

    # An empty tensor still produces a header and round-trips
    empty = torch.empty([0], dtype=torch.uint8, device=dev)
    comp = compress_one(empty)
    out = decompress_one(comp)
    assert comp.numel() > 0
    assert out.numel() == 0
    assert torch.equal(empty, out)
    print(f"  empty tensor -> {comp.numel()} B header, round-trip ok")

    # Byte counts that are not a multiple of 4 are fine on the pointer-batch path
    for n in [1, 2, 3, 5, 101, 1000003]:
        t = torch.randint(0, 65, [n], dtype=torch.uint8, device=dev)
        assert torch.equal(decompress_one(compress_one(t)), t)
    print("  n in {1, 2, 3, 5, 101, 1000003} (not multiple of 4): round-trip ok")

    # Inputs must be contiguous CUDA tensors
    non_contig = torch.randint(0, 65, [4096, 2], dtype=torch.uint8, device=dev)[:, 0]
    assert not non_contig.is_contiguous()
    for bad in [non_contig, torch.zeros([1024], dtype=torch.uint8)]:
        try:
            compress_one(bad)
            raise AssertionError("expected the input to be rejected")
        except RuntimeError:
            pass
    print("  non-contiguous and CPU-only inputs rejected; use .contiguous().cuda()")


def test_uint8_large_tensor_roundtrip():
    """A 64 MB tensor: correctness plus compress / decompress throughput."""
    dev = torch.device("cuda:0")
    temp_mem = make_temp_mem(dev)
    n = 1 << 26
    t = torch.randint(0, 65, [n], dtype=torch.uint8, device=dev)
    print(f"\n[large] {n} B ({n / 1e6:.1f} MB) uint8 tensor")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    comp, sizes, _ = dietgpu.compress_data(COMPRESS_AS_FLOAT, [t], True, temp_mem)
    end.record()
    torch.cuda.synchronize()
    comp_ms = start.elapsed_time(end)

    truncated = comp[0].narrow(0, 0, sizes[0].item()).clone()
    out = torch.empty([n], dtype=torch.uint8, device=dev)

    start.record()
    dietgpu.decompress_data(COMPRESS_AS_FLOAT, [truncated], [out], True, temp_mem)
    end.record()
    torch.cuda.synchronize()
    decomp_ms = start.elapsed_time(end)

    assert torch.equal(t, out)

    print(
        f"  compress   {comp_ms:.3f} ms ({n / 1e9 / (comp_ms * 1e-3):.1f} GB/s), "
        f"{n} B -> {sizes[0].item()} B ({sizes[0].item() / n:.4f}x)"
    )
    print(f"  decompress {decomp_ms:.3f} ms ({n / 1e9 / (decomp_ms * 1e-3):.1f} GB/s)")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("DietGPU requires a CUDA device")

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"torch:  {torch.__version__}")

    for test in [
        test_uint8_simple_api,
        test_uint8_batch_api,
        test_uint8_preallocated_compressed_buffer,
        test_uint8_compression_ratio_by_entropy,
        test_uint8_checksum_detects_corruption,
        test_uint8_shape_is_flattened_to_bytes,
        test_uint8_float8_payload,
        test_uint8_split_size_api,
        test_uint8_edge_cases,
        test_uint8_large_tensor_roundtrip,
    ]:
        test()

    print("\nall uint8 checks passed")


if __name__ == "__main__":
    main()
