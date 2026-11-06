/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cooperative_groups.h>
#include <glog/logging.h>

#include <atomic>
#include <cmath>
#include <memory>
#include <type_traits>
#include <vector>

#include "dietgpu/ans/BatchProvider.cuh"
#include "dietgpu/ans/GpuANSEncode.cuh"
#include "dietgpu/ans/GpuANSUtils.cuh"
#include "dietgpu/ans/GpuChecksum.cuh"
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/float/GpuFloatUtils.cuh"
#include "dietgpu/utils/DeviceDefs.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/StackDeviceMemory.h"
#include "dietgpu/utils/StaticUtils.h"

namespace dietgpu {

// Symbol reader handed to the ANS encoder that re-extracts the compressible
// byte from the original float words, instead of reading it out of an exponent
// array materialized by splitFloat. Extraction is a couple of shifts, and the
// encoder is bound by the serial dependency chain in its ANS state rather than
// by issue rate, so re-deriving the symbol is close to free -- while the buffer
// it replaces is a byte per float word for the whole batch.
//
// Only a win while a float word is 2 bytes: then the encoder reads 2 bytes to
// recover 1, exactly offsetting the 1 byte of writes plus 1 byte of reads that
// staging the exponent stream costs. At 4 bytes per word the extra reads
// outweigh the saved writes, so float32 keeps staging its exponents and uses
// ANSSymbolBytes.
template <FloatType FT>
struct ANSSymbolFloatComp {
    using FTI = FloatTypeInfo<FT>;
    using WordT = typename FTI::WordT;

    const WordT* __restrict__ in;

    static __device__ __forceinline__ ANSSymbolFloatComp make(const void* base,
                                                              uint32_t offset) {
        auto p = (const WordT*)base + offset;

        // float data is only guaranteed float word aligned, not 4 byte aligned
        // as the raw byte path requires
        assert(isPointerAligned(p, sizeof(WordT)));

        return {p};
    }

    __device__ __forceinline__ ANSDecodedT operator[](uint32_t i) const {
        typename FTI::CompT comp;
        typename FTI::NonCompT nonComp;

        // the same split() the staging path used, so the symbols -- and hence
        // the archive -- are identical either way
        FTI::split(in[i], comp, nonComp);

        return comp;
    }
};

template <FloatType FT, int Threads, bool WriteComp>
struct SplitFloatNonAligned {
    static __device__ void split(
        const typename FloatTypeInfo<FT>::WordT* in, uint32_t size,
        typename FloatTypeInfo<FT>::CompT* compOut,
        typename FloatTypeInfo<FT>::NonCompT* nonCompOut,
        uint32_t* warpHistogram) {
        using FTI = FloatTypeInfo<FT>;
        using CompT = typename FTI::CompT;
        using NonCompT = typename FTI::NonCompT;

        for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
             i += gridDim.x * blockDim.x) {
            CompT comp;
            NonCompT nonComp;
            FTI::split(in[i], comp, nonComp);

            atomicAdd(&warpHistogram[comp], 1);

            if constexpr (WriteComp) {
                compOut[i] = comp;
            }
            nonCompOut[i] = nonComp;
        }
    }
};

template <int Threads, bool WriteComp>
struct SplitFloatNonAligned<FloatType::kFloat32, Threads, WriteComp> {
    static __device__ void split(
        const typename FloatTypeInfo<FloatType::kFloat32>::WordT* in,
        uint32_t size,
        typename FloatTypeInfo<FloatType::kFloat32>::CompT* compOut,
        typename FloatTypeInfo<FloatType::kFloat32>::NonCompT* nonCompOut,
        uint32_t* warpHistogram) {
        using FTI = FloatTypeInfo<FloatType::kFloat32>;
        using CompT = typename FTI::CompT;
        using NonCompT = typename FTI::NonCompT;

        // Where the low order 2 bytes are written
        uint16_t* nonComp2Out = (uint16_t*)nonCompOut;

        // Where the high order byte is written
        uint8_t* nonComp1Out = (uint8_t*)(nonComp2Out + roundUp(size, 8));

        for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
             i += gridDim.x * blockDim.x) {
            CompT comp;
            NonCompT nonComp;
            FTI::split(in[i], comp, nonComp);

            nonComp2Out[i] = nonComp & 0xffffU;
            nonComp1Out[i] = nonComp >> 16;

            if constexpr (WriteComp) {
                compOut[i] = comp;
            }

            atomicAdd(&warpHistogram[comp], 1);
        }
    }
};

template <FloatType FT, int Threads, bool WriteComp>
struct SplitFloatAligned16 {
    static __device__ void split(
        const typename FloatTypeInfo<FT>::WordT* __restrict__ in, uint32_t size,
        typename FloatTypeInfo<FT>::CompT* __restrict__ compOut,
        typename FloatTypeInfo<FT>::NonCompT* __restrict__ nonCompOut,
        uint32_t* warpHistogram) {
        using FTI = FloatTypeInfo<FT>;

        using WordT = typename FTI::WordT;
        using CompT = typename FTI::CompT;
        using NonCompT = typename FTI::NonCompT;
        using VecT = typename FTI::VecT;
        using CompVecT = typename FTI::CompVecT;
        using NonCompVecT = typename FTI::NonCompVecT;

        constexpr int kOuterUnroll = 2;
        constexpr int kInnerUnroll = sizeof(VecT) / sizeof(WordT);

        const VecT* inV = (const VecT*)in;
        CompVecT* compOutV = (CompVecT*)compOut;
        NonCompVecT* nonCompOutV = (NonCompVecT*)nonCompOut;

        // Each block handles Threads * kOuterUnroll * kInnerUnroll
        // inputs/outputs at a time, or Threads * kOuterUnroll 16-byte words at
        // a time

        constexpr int kWordsPerBlock = Threads * kOuterUnroll;
        constexpr int kFloatsPerBlock = kWordsPerBlock * kInnerUnroll;
        uint32_t fullBlocks = divDown(size, kFloatsPerBlock);

        // Handle by block
        uint32_t startBlock = blockIdx.x * kWordsPerBlock;
        inV += startBlock + threadIdx.x;
        compOutV += startBlock + threadIdx.x;
        nonCompOutV += startBlock + threadIdx.x;

        for (uint32_t b = blockIdx.x; b < fullBlocks; b += gridDim.x,
                      inV += gridDim.x * kWordsPerBlock,
                      compOutV += gridDim.x * kWordsPerBlock,
                      nonCompOutV += gridDim.x * kWordsPerBlock) {
            VecT v[kOuterUnroll];

#pragma unroll
            for (uint32_t i = 0; i < kOuterUnroll; ++i) {
                v[i] = inV[i * Threads];
            }

            CompVecT compV[kOuterUnroll];
            NonCompVecT nonCompV[kOuterUnroll];

#pragma unroll
            for (uint32_t i = 0; i < kOuterUnroll; ++i) {
#pragma unroll
                for (int j = 0; j < kInnerUnroll; ++j) {
                    CompT comp;
                    NonCompT nonComp;
                    FTI::split(v[i].x[j], comp, nonComp);

                    atomicAdd(&warpHistogram[comp], 1);

                    compV[i].x[j] = comp;
                    nonCompV[i].x[j] = nonComp;
                }
            }

#pragma unroll
            for (uint32_t i = 0; i < kOuterUnroll; ++i) {
                compOutV[i * Threads] = compV[i];
                nonCompOutV[i * Threads] = nonCompV[i];
            }
        }

        // Handle last (partial) block
        for (uint32_t i = fullBlocks * kFloatsPerBlock + blockIdx.x * Threads +
                          threadIdx.x;
             i < size; i += gridDim.x * Threads) {
            CompT comp;
            NonCompT nonComp;
            FTI::split(in[i], comp, nonComp);

            atomicAdd(&warpHistogram[comp], 1);

            if constexpr (WriteComp) {
                compOut[i] = comp;
            }
            nonCompOut[i] = nonComp;
        }
    }
};

// float32 specialization
template <int Threads, bool WriteComp>
struct SplitFloatAligned16<FloatType::kFloat32, Threads, WriteComp> {
    static __device__ void split(
        const typename FloatTypeInfo<
            FloatType::kFloat32>::WordT* __restrict__ in,
        uint32_t size,
        typename FloatTypeInfo<
            FloatType::kFloat32>::CompT* __restrict__ compOut,
        typename FloatTypeInfo<
            FloatType::kFloat32>::NonCompT* __restrict__ nonCompOut,
        uint32_t* warpHistogram) {
        using FTI = FloatTypeInfo<FloatType::kFloat32>;

        using WordT = typename FTI::WordT;
        using CompT = typename FTI::CompT;
        using NonCompT = typename FTI::NonCompT;

        constexpr int kOuterUnroll = 1;
        constexpr int kInnerUnroll = sizeof(uint32x4) / sizeof(uint32_t);

        auto inV = (const uint32x4*)in;
        auto compOutV = (uint8x4*)compOut;

        auto nonCompOut2 = (uint16_t*)nonCompOut;
        auto nonCompOut1 = (uint8_t*)(nonCompOut2 + roundUp(size, 8));

        auto nonCompOutV2 = (uint16x4*)nonCompOut2;
        auto nonCompOutV1 = (uint8x4*)nonCompOut1;

        // Each block handles Threads * kOuterUnroll * kInnerUnroll
        // inputs/outputs at a time, or Threads * kOuterUnroll 16-byte words at
        // a time
        constexpr int kWordsPerBlock = Threads * kOuterUnroll;
        constexpr int kFloatsPerBlock = kWordsPerBlock * kInnerUnroll;
        uint32_t fullBlocks = divDown(size, kFloatsPerBlock);

        // Handle by block
        uint32_t startBlock = blockIdx.x * kWordsPerBlock;
        inV += startBlock + threadIdx.x;
        compOutV += startBlock + threadIdx.x;
        nonCompOutV2 += startBlock + threadIdx.x;
        nonCompOutV1 += startBlock + threadIdx.x;

        for (uint32_t b = blockIdx.x; b < fullBlocks; b += gridDim.x,
                      inV += gridDim.x * kWordsPerBlock,
                      compOutV += gridDim.x * kWordsPerBlock,
                      nonCompOutV2 += gridDim.x * kWordsPerBlock,
                      nonCompOutV1 += gridDim.x * kWordsPerBlock) {
            uint32x4 v[kOuterUnroll];

#pragma unroll
            for (uint32_t i = 0; i < kOuterUnroll; ++i) {
                v[i] = inV[i * Threads];
            }

            uint8x4 compV[kOuterUnroll];
            uint32x4 nonCompV[kOuterUnroll];

#pragma unroll
            for (uint32_t i = 0; i < kOuterUnroll; ++i) {
#pragma unroll
                for (int j = 0; j < kInnerUnroll; ++j) {
                    CompT comp;
                    NonCompT nonComp;
                    FTI::split(v[i].x[j], comp, nonComp);

                    atomicAdd(&warpHistogram[comp], 1);

                    compV[i].x[j] = comp;
                    nonCompV[i].x[j] = nonComp;
                }
            }

#pragma unroll
            for (uint32_t i = 0; i < kOuterUnroll; ++i) {
                compOutV[i * Threads] = compV[i];

                uint16x4 nonCompV2;
                uint8x4 nonCompV1;
                for (int j = 0; j < kInnerUnroll; ++j) {
                    nonCompV2.x[j] = nonCompV[i].x[j] & 0xffffU;
                    nonCompV1.x[j] = nonCompV[i].x[j] >> 16;
                }

                nonCompOutV2[i * Threads] = nonCompV2;
                nonCompOutV1[i * Threads] = nonCompV1;
            }
        }

        // Handle last (partial) block
        for (uint32_t i = fullBlocks * kFloatsPerBlock + blockIdx.x * Threads +
                          threadIdx.x;
             i < size; i += gridDim.x * Threads) {
            CompT comp;
            NonCompT nonComp;
            FTI::split(in[i], comp, nonComp);

            atomicAdd(&warpHistogram[comp], 1);

            compOut[i] = comp;
            nonCompOut2[i] = nonComp & 0xffffU;
            nonCompOut1[i] = nonComp >> 16;
        }
    }
};

// The body of splitFloat, as a device function so that the cooperative
// single-launch compressor below can run it as its own first phase.
//
// It reads its work assignment from blockIdx / gridDim, as do the Split*
// helpers it calls, so a kernel hosting it has to use the same grid shape:
// (chunks, numInBatch) with Threads threads. Its per-warp histogram buckets
// come from the caller, so that a multi-phase kernel can overlay them with the
// scratch of phases a grid barrier separates it from.
template <typename InProvider, typename NonCompProvider, FloatType FT,
          int Threads, bool WriteComp>
__device__ void splitFloatBatchMember(InProvider& inProvider, bool useChecksum,
                                      const uint32_t* __restrict__ checksum,
                                      void* __restrict__ compOut,
                                      uint32_t compOutStride,
                                      NonCompProvider& nonCompProvider,
                                      uint32_t* __restrict__ histogramOut,
                                      uint32_t (*histogram)[kNumSymbols + 1]) {
    using WordT = typename FloatTypeInfo<FT>::WordT;
    using CompT = typename FloatTypeInfo<FT>::CompT;
    using NonCompT = typename FloatTypeInfo<FT>::NonCompT;

    constexpr int kWarps = Threads / kWarpSize;
    static_assert(Threads == kNumSymbols, "");

    auto batch = blockIdx.y;
    auto warpId = threadIdx.x / kWarpSize;

    histogramOut += batch * kNumSymbols;
    checksum += batch;

#pragma unroll
    for (int i = 0; i < kWarps; ++i) {
        histogram[i][threadIdx.x] = 0;
    }

    __syncthreads();

    uint32_t* warpHistogram = histogram[warpId];

    auto curIn = (const WordT*)inProvider.getBatchStart(batch);
    auto headerOut = (GpuFloatHeader*)nonCompProvider.getBatchStart(batch);
    auto curSize = inProvider.getBatchSize(batch);

    // Where the extracted compressible bytes are staged, if they are staged at
    // all. When the ANS encoder re-extracts them from curIn itself there is no
    // exponent buffer, so point at a stand-in: nothing stores through it, but
    // keeping it a valid address leaves the splitters' own address arithmetic
    // well defined instead of computing an offset past a null pointer.
    CompT* curCompOut;
    if constexpr (WriteComp) {
        curCompOut = (CompT*)compOut + compOutStride * batch;
    } else {
        curCompOut = (CompT*)(headerOut + 1);
    }

    // Write size as a header
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        GpuFloatHeader h;
        h.setMagicAndVersion();
        h.size = curSize;
        h.setFloatType(FT);
        h.setUseChecksum(useChecksum);

        if (useChecksum) {
            h.setChecksum(*checksum);
        }

        *headerOut = h;
    }

    auto curNonCompOut = (NonCompT*)(headerOut + 1);

    // How many bytes are before the point where we are 16 byte aligned?
    auto nonAlignedBytes = getAlignmentRoundUp<sizeof(uint4)>(curIn);

    if (nonAlignedBytes > 0) {
        SplitFloatNonAligned<FT, Threads, WriteComp>::split(
            curIn, curSize, curCompOut, curNonCompOut, warpHistogram);
    } else {
        SplitFloatAligned16<FT, Threads, WriteComp>::split(
            curIn, curSize, curCompOut, curNonCompOut, warpHistogram);
    }

    // Accumulate warp histogram data and write into the gmem histogram
    __syncthreads();

    uint32_t sum = histogram[0][threadIdx.x];
#pragma unroll
    for (int j = 1; j < kWarps; ++j) {
        sum += histogram[j][threadIdx.x];
    }

    // The count for the thread's bucket could be 0
    if (sum) {
        atomicAdd(&histogramOut[threadIdx.x], sum);
    }
}

template <typename InProvider, typename NonCompProvider, FloatType FT,
          int Threads, bool WriteComp>
__global__ void splitFloat(InProvider inProvider, bool useChecksum,
                           const uint32_t* __restrict__ checksum,
                           void* __restrict__ compOut, uint32_t compOutStride,
                           NonCompProvider nonCompProvider,
                           uint32_t* __restrict__ histogramOut) {
    constexpr int kWarps = Threads / kWarpSize;

    // +1 in order to force very common symbols that could overlap into
    // different banks between different warps
    __shared__ uint32_t histogram[kWarps][kNumSymbols + 1];

    splitFloatBatchMember<InProvider, NonCompProvider, FT, Threads, WriteComp>(
        inProvider, useChecksum, checksum, compOut, compOutStride,
        nonCompProvider, histogramOut, histogram);
}

// Update the final byte counts for the batch to take into account the
// uncompressed and compressed portions
template <FloatType FT, typename InProvider>
__global__ void incOutputSizes(InProvider inProvider, uint32_t* outSize,
                               uint32_t numInBatch) {
    uint32_t batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch < numInBatch) {
        outSize[batch] +=
            sizeof(GpuFloatHeader) + FloatTypeInfo<FT>::getUncompDataSize(
                                         inProvider.getBatchSize(batch));
    }
}

// Provides the input data to ANS compression
template <typename SizeProvider>
struct FloatANSInProvider {
    using Writer = BatchWriter;

    __host__ FloatANSInProvider(void* ptr_dev, uint32_t stride,
                                SizeProvider& sizeProvider)
        : ptr_dev_(ptr_dev), stride_(stride), sizeProvider_(sizeProvider) {}

    __device__ void* getBatchStart(uint32_t batch) {
        return (uint8_t*)ptr_dev_ + batch * stride_;
    }

    __device__ const void* getBatchStart(uint32_t batch) const {
        return (uint8_t*)ptr_dev_ + batch * stride_;
    }

    __device__ BatchWriter getWriter(uint32_t batch) {
        return BatchWriter(getBatchStart(batch));
    }

    __device__ uint32_t getBatchSize(uint32_t batch) {
        return sizeProvider_.getBatchSize(batch);
    }

    void* ptr_dev_;
    uint32_t stride_;
    SizeProvider sizeProvider_;
};

// Provides the output data to ANS compression
template <FloatType FT, typename OutProvider, typename SizeProvider>
struct FloatANSOutProvider {
    using Writer = BatchWriter;
    using FTI = FloatTypeInfo<FT>;

    __host__ FloatANSOutProvider(OutProvider& outProvider,
                                 SizeProvider& sizeProvider)
        : outProvider_(outProvider), sizeProvider_(sizeProvider) {}

    __device__ void* getBatchStart(uint32_t batch) {
        uint8_t* p = (uint8_t*)outProvider_.getBatchStart(batch);

        // Increment the pointer to past the floating point data
        ((GpuFloatHeader*)p)->checkMagicAndVersion();
        return p + sizeof(GpuFloatHeader) +
               FTI::getUncompDataSize(sizeProvider_.getBatchSize(batch));
    }

    __device__ const void* getBatchStart(uint32_t batch) const {
        const uint8_t* p = (const uint8_t*)outProvider_.getBatchStart(batch);

        // Increment the pointer to past the floating point data
        ((GpuFloatHeader*)p)->checkMagicAndVersion();
        return p + sizeof(GpuFloatHeader) +
               FTI::getUncompDataSize(sizeProvider_.getBatchSize(batch));
    }

    __device__ BatchWriter getWriter(uint32_t batch) {
        return BatchWriter(getBatchStart(batch));
    }

    OutProvider outProvider_;
    SizeProvider sizeProvider_;
};

// Single-launch float compression: split, symbol statistics, per-block encode
// and archive finalization in one cooperative kernel -- the float counterpart
// of ansEncodeBatchCooperative, which is where the reasoning for this being
// opt-in, and what the co-residency cap costs, is written down.
//
// Phase 1 here is splitFloat rather than a bare histogram, because splitting
// also writes each float word's incompressible remainder into the archive. That
// is what makes the histogram of the compressible bytes free: they are already
// in registers on the way past.
template <FloatType FT, typename InProvider, typename OutProvider,
          typename OutProviderANS, int Threads, int ProbBits, int BlockSize>
__global__
__launch_bounds__(Threads, kCoopMinBlocksPerSM) void floatCompressCooperative(
    InProvider inProvider, OutProvider outProvider,
    OutProviderANS outProviderANS, bool useChecksum,
    // not const: checksumSingle accumulates into it with atomicXor. Passing it
    // on to splitFloatBatchMember, which only reads it, decays to const.
    uint32_t* __restrict__ checksum, void* __restrict__ compOut,
    uint32_t compOutStride, uint32_t maxNumCompressedBlocks,
    uint32_t stagingBlockStride, uint8_t* __restrict__ staging,
    ANSEncodeCounters* __restrict__ counters, uint32_t* __restrict__ histogram,
    uint32_t probBits, uint32_t* __restrict__ compressedBytes) {
    using FTI = FloatTypeInfo<FT>;

    // Only a 2 byte word can be re-extracted for less traffic than staging it
    // costs, so float32 stages and its symbols arrive as raw bytes.
    constexpr bool kStageComp = sizeof(typename FTI::WordT) > 2;
    using Symbols =
        std::conditional_t<kStageComp, ANSSymbolBytes, ANSSymbolFloatComp<FT>>;

    auto grid = cooperative_groups::this_grid();

    auto batch = blockIdx.y;
    auto tid = threadIdx.x;

    uint32_t uncompressedWords = inProvider.getBatchSize(batch);
    uint32_t numBlocks = divUp(uncompressedWords, BlockSize);

    // Scratch for every phase, overlaid; see ansEncodeBatchCooperative for why.
    // Phase 0 (checksum) and phase 1 (split) share the `stats` member without
    // conflicting, since the reduce scratch and the histogram buckets are
    // separate fields of it.
    constexpr int kWarps = Threads / kWarpSize;

    __shared__ union CoopScratch {
        struct {
            uint32_t histogram[kWarps][kNumSymbols + 1];
            typename cub::BlockReduce<uint32_t, Threads>::TempStorage reduce;
        } stats;

        struct {
            uint4 lookup[kNumSymbols];
            ANSNormalizeScratch<Threads> normalize;
        } encode;
    } scratch;

    // The float header phase 1 writes carries the checksum of the uncompressed
    // input, so when one is wanted it has to be complete before phase 1 reads
    // it. That costs a barrier ahead of everything else, which is why the
    // checksum option is what makes this path three barriers instead of two.
    // Same argument order as checksumBatch uses on the multi-launch path, so
    // the two paths agree on the value even though that value only covers
    // getBatchSize() bytes.
    if (useChecksum) {
        checksumSingle<Threads>((const uint8_t*)inProvider.getBatchStart(batch),
                                uncompressedWords, checksum + batch,
                                scratch.stats.reduce);

        grid.sync();
    }

    // 1. Split, writing the incompressible remainder and the float header into
    // the archive and counting the compressible bytes on the way past.
    splitFloatBatchMember<InProvider, OutProvider, FT, Threads, kStageComp>(
        inProvider, useChecksum, checksum, compOut, compOutStride, outProvider,
        histogram, scratch.stats.histogram);

    // Nothing may normalize until every CTA's histogram contribution has landed
    grid.sync();

    // 2. Normalize this member's histogram into this CTA's own smem table
    auto& smemLookup = scratch.encode.lookup;

    normalizeProbabilitiesFromHistogram<Threads>(
        histogram + batch * kNumSymbols, uncompressedWords, probBits,
        smemLookup, scratch.encode.normalize);

    // The normalizer's last act is each thread writing its own table entry, and
    // the encode loop below indexes the table by symbol
    __syncthreads();

    // Where this member's ANS archive starts: past the float header and the raw
    // remainder phase 1 wrote, which the grid barrier has made visible.
    auto headerOut = (ANSCoalescedHeader*)outProviderANS.getBatchStart(batch);

    const void* symbolBase;
    if constexpr (kStageComp) {
        symbolBase = (const uint8_t*)compOut + (size_t)compOutStride * batch;
    } else {
        symbolBase = inProvider.getBatchStart(batch);
    }

    // 3. Encode every block of this member and place it in the archive
    int laneId = getLaneId();

    // so we know the block is warp uniform
    int firstWarpInGrid =
        __shfl_sync(0xffffffff, (blockIdx.x * Threads + tid) / kWarpSize, 0);
    auto warpsPerGrid = gridDim.x * Threads / kWarpSize;

    auto stagingBatch =
        staging + batch * (size_t)maxNumCompressedBlocks * stagingBlockStride;

    for (uint32_t block = firstWarpInGrid; block < numBlocks;
         block += warpsPerGrid) {
        ansEncodeAndPlaceBlock<Symbols, ProbBits, BlockSize>(
            laneId, block, numBlocks, uncompressedWords, symbolBase, headerOut,
            stagingBatch, stagingBlockStride, &counters[batch].tailWords,
            smemLookup);
    }

    // 4. Elect the last CTA of this batch member to finish the archive
    __shared__ bool amLastCTA;

    __syncthreads();

    if (tid == 0) {
        // Publish this CTA's reservations and block words before announcing
        // arrival, so the CTA elected below reads the final tail.
        __threadfence();

        amLastCTA = (atomicAdd(&counters[batch].arrivals, 1) == gridDim.x - 1);
    }

    __syncthreads();

    if (amLastCTA) {
        // ANS level checksumming is not allowed in float mode; the float header
        // carries it instead
        ansEncodeWriteArchive<Threads>(
            tid, headerOut, numBlocks, uncompressedWords,
            counters[batch].tailWords, smemLookup, probBits, false, 0,
            compressedBytes ? compressedBytes + batch : nullptr);

        // What was just reported is only the ANS encoded portion; add the float
        // header and the raw remainder, which is what the separate
        // incOutputSizes kernel did on the multi-launch path.
        if (tid == 0 && compressedBytes) {
            compressedBytes[batch] += sizeof(GpuFloatHeader) +
                                      FTI::getUncompDataSize(uncompressedWords);
        }
    }
}

template <typename InProvider, typename OutProvider>
void floatCompressDevice(StackDeviceMemory& res,
                         const FloatCompressConfig& config, uint32_t numInBatch,
                         InProvider& inProvider, uint32_t maxSize,
                         OutProvider& outProvider, uint32_t* outSize_dev,
                         cudaStream_t stream) {
    auto maxUncompressedWords = maxSize / sizeof(ANSDecodedT);
    uint32_t maxNumCompressedBlocks =
        divUp(maxUncompressedWords, kDefaultBlockSize);

    // float32 stages its extracted exponent stream; float16 and bfloat16 do
    // not, because for a 2 byte word the encoder can re-extract the
    // compressible byte from the input for less traffic than writing the
    // exponent array out and reading it back. All rows must be 16 byte aligned.
    uint32_t compRowStride = roundUp(maxSize, sizeof(uint4));
    bool stageComp = (config.floatType == FloatType::kFloat32);

    // Single-launch path, when asked for and when this request can be served
    // cooperatively. See ansEncodeBatchDevice for the same decision on the raw
    // byte path, and for why falling back beats failing a request that the
    // multi-launch path below serves perfectly well.
    if (config.ansConfig.useSingleKernel && numInBatch > 0 &&
        getCooperativeLaunchSupportCurrentDevice()) {
        // This path bypasses ansEncodeBatchDevice, and with it the check that
        // lives there. Without it a request for a precision this build does not
        // instantiate would encode at kANSDefaultProbBits while recording the
        // requested precision in the header -- an archive that decodes to
        // garbage with no error reported.
        CHECK_EQ(config.ansConfig.probBits, kANSDefaultProbBits)
            << "this build of dietgpu only instantiates pdf precision "
            << kANSDefaultProbBits << "; see ANSCodecConfig::probBits";

#define RUN_COOP(FLOAT_TYPE)                                                   \
    do {                                                                       \
        constexpr int kBlock = 256;                                            \
        constexpr bool kStageComp =                                            \
            sizeof(typename FloatTypeInfo<FLOAT_TYPE>::WordT) > 2;             \
                                                                               \
        auto outProviderANS =                                                  \
            FloatANSOutProvider<FLOAT_TYPE, OutProvider, InProvider>(          \
                outProvider, inProvider);                                      \
                                                                               \
        auto kernel =                                                          \
            floatCompressCooperative<FLOAT_TYPE, InProvider, OutProvider,      \
                                     decltype(outProviderANS), kBlock,         \
                                     kANSDefaultProbBits, kDefaultBlockSize>;  \
                                                                               \
        uint32_t maxGrid = getMaxResidentBlocks(kernel, kBlock);               \
                                                                               \
        if (numInBatch <= maxGrid) {                                           \
            uint32_t stagingBlockStride =                                      \
                getMaxBlockSizeCoalesced(kDefaultBlockSize);                   \
                                                                               \
            auto staging_dev = res.alloc<uint8_t>(                             \
                stream, (size_t)numInBatch * maxNumCompressedBlocks *          \
                            stagingBlockStride);                               \
                                                                               \
            auto toCompCoop_dev =                                              \
                kStageComp                                                     \
                    ? res.alloc<uint8_t>(stream, numInBatch * compRowStride)   \
                    : res.alloc<uint8_t>(stream, 0);                           \
                                                                               \
            /* histogram, checksum and counters in one reservation, so that a  \
             */                                                                \
            /* single memset zeroes all three; all of them accumulate */       \
            uint32_t scratchWords = numInBatch * (kNumSymbols + 1 + 2);        \
            auto scratch_dev = res.alloc<uint32_t>(stream, scratchWords);      \
                                                                               \
            CUDA_VERIFY(cudaMemsetAsync(scratch_dev.data(), 0,                 \
                                        sizeof(uint32_t) * scratchWords,       \
                                        stream));                              \
                                                                               \
            uint32_t* histogramArg = scratch_dev.data();                       \
            uint32_t* checksumArg = histogramArg + numInBatch * kNumSymbols;   \
            ANSEncodeCounters* countersArg =                                   \
                (ANSEncodeCounters*)(checksumArg + numInBatch);                \
            uint8_t* stagingArg = staging_dev.data();                          \
            uint8_t* compOutArg = toCompCoop_dev.data();                       \
            uint32_t probBitsArg = config.ansConfig.probBits;                  \
            bool useChecksumArg = config.useChecksum;                          \
                                                                               \
            /* the encode phase is a grid-stride loop, so the grid only has to \
             */                                                                \
            /* be big enough to keep the device busy -- but never so big that  \
             */                                                                \
            /* it stops being co-resident, which the launch would reject */    \
            uint32_t gridX = std::max(                                         \
                std::min(                                                      \
                    divUp((int)maxNumCompressedBlocks, kBlock / kWarpSize),    \
                    (int)std::max(maxGrid / numInBatch, 1U)),                  \
                1);                                                            \
            auto grid = dim3(gridX, numInBatch);                               \
                                                                               \
            void* args[] = {                                                   \
                (void*)&inProvider,         (void*)&outProvider,               \
                (void*)&outProviderANS,     (void*)&useChecksumArg,            \
                (void*)&checksumArg,        (void*)&compOutArg,                \
                (void*)&compRowStride,      (void*)&maxNumCompressedBlocks,    \
                (void*)&stagingBlockStride, (void*)&stagingArg,                \
                (void*)&countersArg,        (void*)&histogramArg,              \
                (void*)&probBitsArg,        (void*)&outSize_dev};              \
                                                                               \
            CUDA_VERIFY(cudaLaunchCooperativeKernel(                           \
                (void*)kernel, grid, dim3(kBlock), args, 0, stream));          \
                                                                               \
            CUDA_TEST_ERROR();                                                 \
            return;                                                            \
        }                                                                      \
    } while (false)

        switch (config.floatType) {
            case FloatType::kFloat16:
                RUN_COOP(FloatType::kFloat16);
                break;
            case FloatType::kBFloat16:
                RUN_COOP(FloatType::kBFloat16);
                break;
            case FloatType::kFloat32:
                RUN_COOP(FloatType::kFloat32);
                break;
            default:
                CHECK(false);
                break;
        }

#undef RUN_COOP

        // Reaching here means the batch has more members than this device can
        // keep co-resident. The flag is opt-in, so ignoring it silently would
        // leave the caller believing compression ran in a single launch.
        static std::atomic<bool> warnedCoopFallback{false};

        if (!warnedCoopFallback.exchange(true)) {
            LOG(WARNING)
                << "FloatCompressConfig::ansConfig.useSingleKernel is set, but "
                   "a "
                   "batch of "
                << numInBatch
                << " members exceeds the co-resident blocks this device allows "
                   "for a cooperative launch; compression will keep using the "
                   "multi-launch path";
        }
    }

    // Compute checksum on input data (optional)
    auto checksum_dev = res.alloc<uint32_t>(stream, numInBatch);

    // not allowed in float mode
    assert(!config.ansConfig.useChecksum);

    if (config.useChecksum) {
        checksumBatch(numInBatch, inProvider, checksum_dev.data(), stream);
    }

    // Staging for the extracted exponent stream, allocated only for the float
    // type that needs one -- see compRowStride / stageComp above, which the
    // single-launch path also uses.
    auto toComp_dev =
        stageComp ? res.alloc<uint8_t>(stream, numInBatch * compRowStride)
                  : res.alloc<uint8_t>(stream, 0);

    // We calculate a histogram of the symbols to be compressed as part of
    // extracting the compressible symbol from the float
    auto histogram_dev = res.alloc<uint32_t>(stream, numInBatch * kNumSymbols);

    // zero out buckets before proceeding, as we aggregate with atomic adds
    CUDA_VERIFY(cudaMemsetAsync(histogram_dev.data(), 0,
                                sizeof(uint32_t) * numInBatch * kNumSymbols,
                                stream));

#define RUN_SPLIT(FLOAT_TYPE)                                                 \
    do {                                                                      \
        constexpr int kBlock = 256;                                           \
        constexpr bool kStageComp =                                           \
            sizeof(typename FloatTypeInfo<FLOAT_TYPE>::WordT) > 2;            \
        uint32_t maxGrid =                                                    \
            getMaxResidentBlocks(splitFloat<InProvider, OutProvider,          \
                                            FLOAT_TYPE, kBlock, kStageComp>,  \
                                 kBlock);                                     \
        uint32_t perBatchGrid = 4 * divUp(maxGrid, numInBatch);               \
        auto grid = dim3(perBatchGrid, numInBatch);                           \
                                                                              \
        splitFloat<InProvider, OutProvider, FLOAT_TYPE, kBlock, kStageComp>   \
            <<<grid, kBlock, 0, stream>>>(inProvider, config.useChecksum,     \
                                          checksum_dev.data(),                \
                                          toComp_dev.data(), compRowStride,   \
                                          outProvider, histogram_dev.data()); \
    } while (false)

    switch (config.floatType) {
        case FloatType::kFloat16:
            RUN_SPLIT(FloatType::kFloat16);
            break;
        case FloatType::kBFloat16:
            RUN_SPLIT(FloatType::kBFloat16);
            break;
        case FloatType::kFloat32:
            RUN_SPLIT(FloatType::kFloat32);
            break;
        default:
            assert(false);
            break;
    }

#undef RUN_SPLIT

        // outSize as reported by ansEncode is just the ANS-encoded portion of
        // the data. We need to increment the sizes by the uncompressed portion
        // (header plus uncompressed float data) with incOutputSizes.
        //
        // For a 2 byte float word the encoder is handed the original float data
        // along with ANSSymbolFloatComp, so it re-derives each compressible
        // byte with the same split() the staging path used: the symbol stream,
        // and therefore the archive, is identical either way. float32 keeps
        // reading the staged exponent array through FloatANSInProvider.
#define RUN_ANS(FT)                                                            \
    do {                                                                       \
        using FTI = FloatTypeInfo<FT>;                                         \
                                                                               \
        auto outProviderANS =                                                  \
            FloatANSOutProvider<FT, OutProvider, InProvider>(outProvider,      \
                                                             inProvider);      \
                                                                               \
        if constexpr (sizeof(typename FTI::WordT) == 2) {                      \
            ansEncodeBatchDevice<InProvider, decltype(outProviderANS),         \
                                 ANSSymbolFloatComp<FT>>(                      \
                res, config.ansConfig, numInBatch, inProvider,                 \
                histogram_dev.data(), maxSize, outProviderANS, outSize_dev,    \
                stream);                                                       \
        } else {                                                               \
            auto inProviderANS = FloatANSInProvider<InProvider>(               \
                toComp_dev.data(), compRowStride, inProvider);                 \
                                                                               \
            ansEncodeBatchDevice(res, config.ansConfig, numInBatch,            \
                                 inProviderANS, histogram_dev.data(), maxSize, \
                                 outProviderANS, outSize_dev, stream);         \
        }                                                                      \
                                                                               \
        incOutputSizes<FT><<<divUp(numInBatch, 128), 128, 0, stream>>>(        \
            inProvider, outSize_dev, numInBatch);                              \
                                                                               \
    } while (false)

    // We have written the non-compressed portions of the floats into the
    // output, along with a header that indicates how many floats there are. For
    // compression, we need to increment the address in which the compressed
    // outputs are written.

    switch (config.floatType) {
        case FloatType::kFloat16:
            RUN_ANS(FloatType::kFloat16);
            break;
        case FloatType::kBFloat16:
            RUN_ANS(FloatType::kBFloat16);
            break;
        case FloatType::kFloat32:
            RUN_ANS(FloatType::kFloat32);
            break;
        default:
            assert(false);
            break;
    }

#undef RUN_ANS

    CUDA_TEST_ERROR();
}

}  // namespace dietgpu
