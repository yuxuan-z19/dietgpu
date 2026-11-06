/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <glog/logging.h>

#include <cmath>
#include <cub/block/block_scan.cuh>
#include <memory>
#include <sstream>
#include <vector>

#include "dietgpu/ans/BatchProvider.cuh"
#include "dietgpu/ans/GpuANSCodec.h"
#include "dietgpu/ans/GpuANSInfo.cuh"
#include "dietgpu/ans/GpuANSUtils.cuh"
#include "dietgpu/ans/GpuChecksum.cuh"
#include "dietgpu/utils/DeviceDefs.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/PtxUtils.cuh"
#include "dietgpu/utils/StackDeviceMemory.h"
#include "dietgpu/utils/StaticUtils.h"

namespace dietgpu {

using TableT = uint32_t;

// We are limited to 11 bits of probability resolution
// (worst case, prec = 12, pdf == 2^12, single symbol. 2^12 cannot be
// represented in 12 bits)
inline __device__ TableT packDecodeLookup(uint32_t sym, uint32_t pdf,
                                          uint32_t cdf) {
    static_assert(sizeof(ANSDecodedT) == 1, "");
    // [31:20] cdf
    // [19:8] pdf
    // [7:0] symbol
    return (cdf << 20) | (pdf << 8) | sym;
}

inline __device__ void unpackDecodeLookup(TableT v, uint32_t& sym,
                                          uint32_t& pdf, uint32_t& cdf) {
    // [31:20] cdf
    // [19:8] pdf
    // [7:0] symbol
    sym = v & 0xffU;
    v >>= 8;
    pdf = v & 0xfffU;
    v >>= 12;
    cdf = v;
}

template <int ProbBits>
__device__ void decodeOneWarp(
    ANSStateT& state,

    // Start offset where this warp is reading from the
    // compressed input. As a variable number of lanes
    // wish to read from the compressed offset each
    // iteration, this offset upon calling is one after
    // the last offset, if any, this warp will be reading rom.
    uint32_t compressedOffset,

    const ANSEncodedT* __restrict__ in,

    // Shared memory LUTs
    const TableT* lookup,

    // Output: number of words read from compressed input
    uint32_t& outNumRead,

    // Output: decoded symbol for this iteration
    ANSDecodedT& outSym) {
    constexpr ANSStateT StateMask = (ANSStateT(1) << ProbBits) - ANSStateT(1);

    auto s_bar = state & StateMask;

    uint32_t sym;
    uint32_t pdf;
    uint32_t sMinusCdf;
    unpackDecodeLookup(lookup[s_bar], sym, pdf, sMinusCdf);

    // We always write a decoded value
    outSym = sym;
    state = pdf * (state >> ProbBits) + ANSStateT(sMinusCdf);

    // We only sometimes read a new encoded value
    bool read = state < kANSMinState;
    auto vote = __ballot_sync(0xffffffff, read);
    // We are reading in the same order as we wrote, except by decrementing from
    // compressedOffset, so we need to count down from the highest lane in the
    // warp
    auto prefix = __popc(vote & getLaneMaskGe());

    if (read) {
        auto v = in[-prefix];
        state = (state << kANSEncodedBits) + ANSStateT(v);
    }

    // how many values we actually read from the compressed input
    outNumRead = __popc(vote);
}

template <int ProbBits>
__device__ void decodeOnePartialWarp(
    bool valid, ANSStateT& state,

    // Start offset where this warp is reading from the
    // compressed input. As a variable number of lanes
    // wish to read from the compressed offset each
    // iteration, this offset upon calling is one after
    // the last offset, if any, this warp will be reading rom.
    uint32_t compressedOffset,

    const ANSEncodedT* __restrict__ in,

    // Shared memory LUTs
    const TableT* lookup,

    // Output: number of words read from compressed input
    uint32_t& outNumRead,

    // Output: decoded symbol for this iteration (only if valid)
    ANSDecodedT& outSym) {
    constexpr ANSStateT StateMask = (ANSStateT(1) << ProbBits) - ANSStateT(1);

    auto s_bar = state & StateMask;

    uint32_t sym;
    uint32_t pdf;
    uint32_t sMinusCdf;
    unpackDecodeLookup(lookup[s_bar], sym, pdf, sMinusCdf);

    if (valid) {
        outSym = sym;
        state = pdf * (state >> ProbBits) + ANSStateT(sMinusCdf);
    }

    // We only sometimes read a new encoded value
    bool read = valid && (state < kANSMinState);
    auto vote = __ballot_sync(0xffffffff, read);
    // We are reading in the same order as we wrote, except by decrementing from
    // compressedOffset, so we need to count down from the highest lane in the
    // warp
    auto prefix = __popc(vote & getLaneMaskGe());

    if (read) {
        auto v = in[-prefix];
        state = (state << kANSEncodedBits) + ANSStateT(v);
    }

    // how many values we actually read from the compressed input
    outNumRead = __popc(vote);
}

template <typename Writer, int ProbBits>
__device__ void ansDecodeWarpBlock(int laneId, ANSStateT state,
                                   uint32_t uncompressedWords,
                                   uint32_t compressedWords,
                                   const ANSEncodedT* __restrict__ in,
                                   Writer& writer,
                                   const TableT* __restrict__ table) {
    // The compressed input may not be a whole multiple of a warp.
    // In this case, only the lanes that cover the remainder could have read a
    // value in the input, and thus, only they can write a value in the output.
    // We handle this partial data first.
    uint32_t remainder = uncompressedWords % kWarpSize;

    // A fixed number of uncompressed elements are written each iteration
    int uncompressedOffset = uncompressedWords - remainder;

    // A variable number of compressed elements are read each iteration
    uint32_t compressedOffset = compressedWords;

    in += compressedOffset;

    // Partial warp handling the end of the data
    if (remainder) {
        bool valid = laneId < remainder;

        uint32_t numCompressedRead;
        ANSDecodedT sym;

        decodeOnePartialWarp<ProbBits>(valid, state, compressedOffset, in,
                                       table, numCompressedRead, sym);

        if (valid) {
            writer.write(uncompressedOffset + laneId, sym);
        }

        in -= numCompressedRead;
    }

    // Full warp handling
    while (uncompressedOffset > 0) {
        uncompressedOffset -= kWarpSize;

        uint32_t numCompressedRead;
        ANSDecodedT sym;

        decodeOneWarp<ProbBits>(state, compressedOffset, in, table,
                                numCompressedRead, sym);

        writer.write(uncompressedOffset + laneId, sym);

        in -= numCompressedRead;
    }
}

// Decode a single, full sized block of data.
//
// A vectorized variant -- ANSDecodedTx4 loads, a prefetch.global.L1 issued
// kCacheLinesAhead iterations early, and a preload/writeVec pair on the writer
// -- was tried and measured no faster than the scalar loop below
// (kCacheLinesAhead 2/3/4 gave 252.16/246.62/254.91 us), so it was deleted
// rather than kept behind a compile-time switch that only ever had one live
// setting.
template <typename Writer, int ProbBits, int BlockSize>
struct ANSDecodeWarpFullBlock {
    static __device__ void decode(int laneId, ANSStateT state,
                                  uint32_t compressedWords,
                                  const ANSEncodedT* __restrict__ in,
                                  Writer& writer,
                                  const TableT* __restrict__ table) {
        in += compressedWords;

        for (int i = BlockSize - kWarpSize + laneId; i >= 0; i -= kWarpSize) {
            ANSDecodedT sym;
            uint32_t numCompressedRead;

            decodeOneWarp<ProbBits>(state, compressedWords, in, table,
                                    numCompressedRead, sym);

            in -= numCompressedRead;

            writer.write(i, sym);
        }
    }
};

template <typename InProvider, typename OutProvider, int Threads, int ProbBits,
          int BlockSize>
__global__ __launch_bounds__(Threads) void ansDecodeKernel(
    InProvider inProvider, OutProvider outProvider,
    uint8_t* __restrict__ outSuccess, uint32_t* __restrict__ outSize) {
    auto tid = threadIdx.x;
    auto batch = blockIdx.y;

    // Interpret header as uint4
    auto headerIn = (const ANSCoalescedHeader*)inProvider.getBatchStart(batch);
    headerIn->checkMagicAndVersion();

    auto header = *headerIn;
    auto numBlocks = header.getNumBlocks();
    auto totalUncompressedWords = header.getTotalUncompressedWords();

    // Is the data what we expect? The caller selects this instantiation from
    // config.probBits, while the archive records the precision it was actually
    // written with. A mismatch means the wrong instantiation is decoding this
    // batch -- assert() alone does not catch that, because NDEBUG compiles it
    // out of Release builds and the result would be silent garbage, so fold it
    // into the success flag that the caller can actually observe.
    auto precisionMatches = (ProbBits == header.getProbBits());
    assert(precisionMatches);

    // Do we have enough space for the decompressed data?
    auto uncompressedBytes = totalUncompressedWords * sizeof(ANSDecodedT);
    bool success = precisionMatches &&
                   outProvider.getBatchSize(batch) >= uncompressedBytes;

    if (blockIdx.x == 0 && tid == 0) {
        if (outSuccess) {
            outSuccess[batch] = success;
        }

        if (outSize) {
            outSize[batch] = uncompressedBytes;
        }
    }

    if (!success) {
        return;
    }

    // warp id taking into account warps in the current block
    // do this so the compiler knows it is warp uniform
    int globalWarpId =
        __shfl_sync(0xffffffff, (blockIdx.x * blockDim.x + tid) / kWarpSize, 0);

    auto warpsPerGrid = gridDim.x * Threads / kWarpSize;
    int laneId = getLaneId();

    // The grid is deliberately oversubscribed (~4x what the device can keep
    // resident) to cover tail effects and uneven provisioning across the batch,
    // so most CTAs have nothing to decode and can leave before building the
    // lookup table below. The test has to be CTA uniform: warps within a CTA
    // are consecutive and the decode loop steps by the whole grid, so the CTA
    // has work iff its first warp does -- testing globalWarpId instead would
    // return from some warps only and hang the rest on the __syncthreads()
    // below.
    uint32_t firstWarpInCTA = (blockIdx.x * Threads) / kWarpSize;

    if (firstWarpInCTA >= numBlocks) {
        return;
    }

    // Build the rANS decode table in smem straight from the archive's pdf.
    //
    // This used to be its own kernel, which expanded each archive's 256 symbol
    // probabilities into a 1 << ProbBits bucket table in gmem that every CTA
    // here then copied into smem. Deriving it in place removes that launch,
    // that per-batch buffer and a 4 KiB gmem read per CTA; what it costs is
    // redoing the expansion once per CTA rather than once per batch member,
    // which is cheap next to decoding a 4 KiB block and runs concurrently
    // across CTAs.
    constexpr int kBuckets = 1 << ProbBits;
    __shared__ TableT lookup[kBuckets];

    {
        // cub's array scan wants a contiguous blocked arrangement, so thread
        // tid owns symbols tid * kSymPerThread + i
        constexpr int kSymPerThread = kNumSymbols / Threads;
        static_assert(kNumSymbols % Threads == 0, "");
        static_assert(kSymPerThread >= 1, "");

        auto probs = headerIn->getSymbolProbs();

        uint32_t pdf[kSymPerThread];
        uint32_t cdf[kSymPerThread];

#pragma unroll
        for (int i = 0; i < kSymPerThread; ++i) {
            pdf[i] = probs[tid * kSymPerThread + i];
        }

        // Get the CDF from the PDF
        using BlockScan = cub::BlockScan<uint32_t, Threads>;
        __shared__ typename BlockScan::TempStorage tempStorage;

        uint32_t total = 0;
        BlockScan(tempStorage).ExclusiveSum(pdf, cdf, total);

        // should be a power of 2
        assert(uint32_t(1 << ProbBits) == total);

        // Publish the CDF indexed by symbol, with a sentinel, so that any
        // bucket can be mapped back to the symbol that owns it.
        __shared__ uint32_t smemCdf[kNumSymbols + 1];

#pragma unroll
        for (int i = 0; i < kSymPerThread; ++i) {
            smemCdf[tid * kSymPerThread + i] = cdf[i];
        }

        if (tid == 0) {
            smemCdf[kNumSymbols] = 1 << ProbBits;
        }

        __syncthreads();

        // Fill the bucket table with each thread owning its own run of buckets,
        // rather than one warp walking the 256 symbols in turn. Bucket j
        // belongs to the *largest* symbol s with smemCdf[s] <= j: taking the
        // largest skips symbols whose pdf is zero, and for that s the pdf
        // smemCdf[s + 1] - smemCdf[s] is necessarily non-zero, since otherwise
        // s would not have been the largest.
        //
        // The searches are log2(kNumSymbols) deep and independent across
        // threads, so the cost does not depend on how the probability mass is
        // spread. Both alternatives do: a symbol-serial loop is long when the
        // pdf is peaked, and one thread per symbol serializes an archive
        // holding a single distinct value -- which is exactly what all-zero
        // input produces.
        constexpr int kBucketsPerThread = kBuckets / Threads;
        static_assert(kBuckets % Threads == 0, "");

#pragma unroll
        for (int b = 0; b < kBucketsPerThread; ++b) {
            uint32_t bucket = b * Threads + tid;

            // largest lo in [0, kNumSymbols) with smemCdf[lo] <= bucket,
            // maintaining the invariant smemCdf[lo] <= bucket < smemCdf[hi]
            uint32_t lo = 0;
            uint32_t hi = kNumSymbols;

            while (hi - lo > 1) {
                uint32_t mid = (lo + hi) / 2;

                if (smemCdf[mid] <= bucket) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }

            auto symBegin = smemCdf[lo];
            auto symPdf = smemCdf[lo + 1] - symBegin;

            // within-bucket cdf is bucket - symBegin
            lookup[bucket] = packDecodeLookup(lo, symPdf, bucket - symBegin);
        }
    }

    __syncthreads();

    auto writer = outProvider.getWriter(batch);

    for (int block = globalWarpId; block < numBlocks; block += warpsPerGrid) {
        // Load state
        ANSStateT state = headerIn->getWarpStates()[block].warpState[laneId];

        // Load per-block size data
        auto blockWords = headerIn->getBlockWords(numBlocks)[block];
        uint32_t uncompressedWords = (blockWords.x >> 16);
        uint32_t compressedWords = (blockWords.x & 0xffff);
        uint32_t blockCompressedWordStart = blockWords.y;

        // Get block addresses for encoded/decoded data
        auto blockDataIn =
            headerIn->getBlockDataStart(numBlocks) + blockCompressedWordStart;

        writer.setBlock(block);

        using Writer = typename OutProvider::Writer;
        if (uncompressedWords == BlockSize) {
            ANSDecodeWarpFullBlock<Writer, ProbBits, BlockSize>::decode(
                laneId, state, compressedWords, blockDataIn, writer, lookup);
        } else {
            ansDecodeWarpBlock<Writer, ProbBits>(
                laneId, state, uncompressedWords, compressedWords, blockDataIn,
                writer, lookup);
        }
    }
}

template <typename InProvider, typename OutProvider>
ANSDecodeStatus ansDecodeBatch(
    StackDeviceMemory& res, const ANSCodecConfig& config, uint32_t numInBatch,
    const InProvider& inProvider, OutProvider& outProvider,
    uint8_t* outSuccess_dev, uint32_t* outSize_dev, cudaStream_t stream) {
    // Only kANSDefaultProbBits is instantiated; see the matching check in
    // ansEncodeBatchDevice. Done first so nothing is allocated for a request
    // this build cannot serve.
    CHECK_EQ(config.probBits, kANSDefaultProbBits)
        << "this build of dietgpu only instantiates pdf precision "
        << kANSDefaultProbBits << "; see ANSCodecConfig::probBits";

    // Perform decoding. The rANS lookup table is built in smem by the kernel
    // itself from each archive's pdf, so there is no table buffer to allocate
    // and no separate expansion launch ahead of this one.
    {
        // FIXME: We have no idea how large the decompression job is, as the
        // relevant information is on the device.
        // Just launch a grid that is sufficiently large enough to saturate the
        // GPU; blocks will exit if there isn't enough work, or will loop if
        // there is more work. We aim for a grid >4x larger than what the device
        // can sustain, to help cover up tail effects and unequal provisioning
        // across the batch
        constexpr int kThreads = 256;
        uint32_t maxGrid = getMaxResidentBlocks(
            ansDecodeKernel<InProvider, OutProvider, kThreads,
                            kANSDefaultProbBits, kDefaultBlockSize>,
            kThreads);
        uint32_t perBatchGrid = divUp(maxGrid, numInBatch) * 4;
        auto grid = dim3(perBatchGrid, numInBatch);

        ansDecodeKernel<InProvider, OutProvider, kThreads, kANSDefaultProbBits,
                        kDefaultBlockSize><<<grid, kThreads, 0, stream>>>(
            inProvider, outProvider, outSuccess_dev, outSize_dev);
    }

    ANSDecodeStatus status;

    // Perform optional checksum, if desired
    if (config.useChecksum) {
        auto checksum_dev = res.alloc<uint32_t>(stream, numInBatch);
        auto sizes_dev = res.alloc<uint32_t>(stream, numInBatch);
        auto archiveChecksum_dev = res.alloc<uint32_t>(stream, numInBatch);

        // Checksum the output data
        checksumBatch(numInBatch, outProvider, checksum_dev.data(), stream);

        // Get prior checksum from the ANS headers
        ansGetCompressedInfo(inProvider, numInBatch, sizes_dev.data(),
                             archiveChecksum_dev.data(), stream);

        // Compare against previously seen checksums on the host
        auto sizes = sizes_dev.copyToHost(stream);
        auto newChecksums = checksum_dev.copyToHost(stream);
        auto oldChecksums = archiveChecksum_dev.copyToHost(stream);

        std::stringstream errStr;

        for (int i = 0; i < numInBatch; ++i) {
            if (oldChecksums[i] != newChecksums[i]) {
                status.error = ANSDecodeError::ChecksumMismatch;

                errStr << "Checksum mismatch in batch member " << i
                       << ": expected checksum " << std::hex << oldChecksums[i]
                       << " got " << newChecksums[i] << "\n";
                status.errorInfo.push_back(std::make_pair(i, errStr.str()));
            }
        }
    }

    CUDA_TEST_ERROR();

    return status;
}

}  // namespace dietgpu
