/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cooperative_groups.h>
#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "dietgpu/ans/GpuANSCodec.h"
#include "dietgpu/ans/GpuANSStatistics.cuh"
#include "dietgpu/ans/GpuANSUtils.cuh"
#include "dietgpu/ans/GpuChecksum.cuh"
#include "dietgpu/utils/DeviceDefs.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/PtxUtils.cuh"
#include "dietgpu/utils/StackDeviceMemory.h"
#include "dietgpu/utils/StaticUtils.h"

namespace dietgpu {

// maximum raw compressed data block size in bytes
constexpr __host__ __device__ uint32_t
getRawCompBlockMaxSize(uint32_t uncompressedBlockBytes) {
    // (an estimate from zstd)
    return roundUp(uncompressedBlockBytes + (uncompressedBlockBytes / 4),
                   kBlockAlignment);
}

inline uint32_t getMaxBlockSizeCoalesced(uint32_t uncompressedBlockBytes) {
    return getRawCompBlockMaxSize(uncompressedBlockBytes);
}

// Returns number of values written to the compressed output
// Assumes all lanes in the warp are presented valid input symbols
template <int ProbBits>
__device__ __forceinline__ uint32_t encodeOneWarp(
    ANSStateT& state, ANSDecodedT sym, uint32_t outOffset,
    ANSEncodedT* __restrict__ out, const uint4* __restrict__ smemLookup) {
    auto lookup = smemLookup[sym];

    uint32_t pdf = lookup.x;
    uint32_t cdf = lookup.y;
    uint32_t div_m1 = lookup.z;
    uint32_t div_shift = lookup.w;

    constexpr ANSStateT kStateCheckMul = 1 << (kANSStateBits - ProbBits);

    ANSStateT maxStateCheck = pdf * kStateCheckMul;
    bool write = (state >= maxStateCheck);

    auto vote = __ballot_sync(0xffffffff, write);
    auto prefix = __popc(vote & getLaneMaskLt());

    // Some lanes wish to write out their data
    if (write) {
        out[outOffset + prefix] = state & kANSEncodedMask;
        state >>= kANSEncodedBits;
    }

    constexpr uint32_t kProbBitsMul = 1 << ProbBits;

    uint32_t t = __umulhi(state, div_m1);
    // We prevent addition overflow here by restricting `state` to < 2^31
    // (kANSStateBits)
    uint32_t div = (t + state) >> div_shift;
    auto mod = state - (div * pdf);

    // calculating ((state / pdf) << ProbBits) + (state % pdf) + cdf
    state = div * kProbBitsMul + mod + cdf;

    // how many values we actually write to the compressed output
    return __popc(vote);
}

// Returns number of values written to the compressed output
// Assumes only some lanes in the warp are presented valid input symbols
template <int ProbBits>
__device__ __forceinline__ uint32_t encodeOnePartialWarp(
    // true for the lanes in the warp for which data read is valid
    bool valid, ANSStateT& state, ANSDecodedT sym, uint32_t outOffset,
    ANSEncodedT* __restrict__ out, const uint4* __restrict__ smemLookup) {
    auto lookup = smemLookup[sym];

    uint32_t pdf = lookup.x;
    uint32_t cdf = lookup.y;
    uint32_t div_m1 = lookup.z;
    uint32_t div_shift = lookup.w;

    constexpr ANSStateT kStateCheckMul = 1 << (kANSStateBits - ProbBits);

    ANSStateT maxStateCheck = pdf * kStateCheckMul;
    bool write = valid && (state >= maxStateCheck);

    auto vote = __ballot_sync(0xffffffff, write);
    auto prefix = __popc(vote & getLaneMaskLt());

    // Some lanes wish to write out their data
    if (write) {
        out[outOffset + prefix] = state & kANSEncodedMask;
        state >>= kANSEncodedBits;
    }

    uint32_t t = __umulhi(state, div_m1);
    // We prevent addition overflow here by restricting `state` to < 2^31
    // (kANSStateBits)
    uint32_t div = (t + state) >> div_shift;
    auto mod = state - (div * pdf);

    // calculating ((state / pdf) << ProbBits) + (state % pdf) + cdf
    constexpr uint32_t kProbBitsMul = 1 << ProbBits;
    state = valid ? div * kProbBitsMul + mod + cdf : state;

    // how many values we actually write to the compressed output
    return __popc(vote);
}

// Where a warp reads the symbols it encodes. The byte path reads them straight
// out of memory; the float path supplies its own reader that re-extracts the
// compressible byte from the original float words, so that the extracted stream
// never has to be materialized in a temporary buffer. Readers are passed by
// value and hold a single pointer, so indexing through one compiles to the same
// load as indexing the raw pointer did.
struct ANSSymbolBytes {
    const ANSDecodedT* __restrict__ in;

    static __device__ __forceinline__ ANSSymbolBytes make(const void* base,
                                                          uint32_t offset) {
        auto p = (const ANSDecodedT*)base + offset;

        // all input blocks must meet alignment requirements
        assert(isPointerAligned(p, kANSRequiredAlignment));

        return {p};
    }

    __device__ __forceinline__ ANSDecodedT operator[](uint32_t i) const {
        return in[i];
    }
};

// Fully encode a single block of data. Each lane's final ANS state goes to
// stateOut and the encoded words go to wordsOut; the two are separate because
// the state belongs in the archive header while the words are staged elsewhere
// until their final home is known.
// Returns the number of compressed words (ANSEncodedT) written
template <int ProbBits, typename Symbols>
__device__ uint32_t ansEncodeWarpBlock(
    // Current lane ID in the warp
    uint32_t laneId,
    // Input symbols for this block
    Symbols syms,
    // Number of symbols in this block
    uint32_t inWords,
    // encoded table in smem
    const uint4* __restrict__ table,
    // Output: this warp's final ANS state
    ANSWarpState* __restrict__ stateOut,
    // Output: the encoded words
    ANSEncodedT* __restrict__ wordsOut) {
    // Start state value for this warp
    ANSStateT state = kANSStartState;

    uint32_t inOffset = laneId;
    uint32_t outOffset = 0;

    constexpr int kUnroll = 8;

    // Unrolled iterations
    uint32_t limit = roundDown(inWords, kWarpSize * kUnroll);
    {
        ANSDecodedT sym[kUnroll];

        for (; inOffset < limit; inOffset += kWarpSize * kUnroll) {
#pragma unroll
            for (int j = 0; j < kUnroll; ++j) {
                sym[j] = syms[inOffset + j * kWarpSize];
            }

#pragma unroll
            for (int j = 0; j < kUnroll; ++j) {
                outOffset += encodeOneWarp<ProbBits>(state, sym[j], outOffset,
                                                     wordsOut, table);
            }
        }
    }

    if (limit != inWords) {
        // Remainder iterations
        limit = roundDown(inWords, kWarpSize);

        // Whole warp iterations
        for (; inOffset < limit; inOffset += kWarpSize) {
            ANSDecodedT sym = syms[inOffset];

            outOffset +=
                encodeOneWarp<ProbBits>(state, sym, outOffset, wordsOut, table);
        }

        // Partial warp iteration
        if (limit != inWords) {
            // Last iteration may not be a full warp
            bool valid = inOffset < inWords;
            ANSDecodedT sym = valid ? syms[inOffset] : ANSDecodedT(0);

            outOffset += encodeOnePartialWarp<ProbBits>(
                valid, state, sym, outOffset, wordsOut, table);
        }
    }

    // Write final state (aligned addresses)
    stateOut->warpState[laneId] = state;

    // Number of compressed words written
    return outOffset;
}

// Fully encode a single, full sized block of data.
// Returns the number of compressed words (ANSEncodedT) written
//
// Kept separate from ansEncodeWarpBlock even though it just forwards to it:
// passing BlockSize as a compile time constant lets the compiler fold the loop
// bound and drop the remainder handling entirely, which the variable sized
// version cannot do.
//
// A vectorized variant -- each lane loading 4 input bytes at a time and the
// warp compressing them interleaved -- was tried and measured no faster than
// the scalar loop below (kUnroll 4 with an inner unroll of 2: 164.93 us;
// kUnroll 8 with no inner unroll: 161.86 us), so it was deleted rather than
// kept behind a compile-time switch that only ever had one live setting.
template <int ProbBits, int BlockSize, typename Symbols>
struct ANSEncodeWarpFullBlock {
    static __device__ uint32_t encode(
        // Current lane ID in the warp
        uint32_t laneId,
        // Input symbols for this block
        Symbols syms,
        // encoded table in smem
        const uint4* __restrict__ table,
        // Output: this warp's final ANS state
        ANSWarpState* __restrict__ stateOut,
        // Output: the encoded words
        ANSEncodedT* __restrict__ wordsOut) {
        // Just use the normal implementation
        return ansEncodeWarpBlock<ProbBits, Symbols>(laneId, syms, BlockSize,
                                                     table, stateOut, wordsOut);
    }
};

// Per-batch encode bookkeeping, in gmem for the lifetime of the fused encode
// kernel. Zeroed before the launch.
struct ANSEncodeCounters {
    // Running total of compressed words (ANSEncodedT) reserved so far. Each
    // warp reserves its own block's aligned length out of this, which both
    // assigns the block its final home and -- once every warp has reserved --
    // gives the archive's total compressed length. That replaces the exclusive
    // prefix sum over per-block lengths (three launches) that the coalescing
    // pass used to need in order to know where each block went.
    uint32_t tailWords;

    // How many CTAs of this batch member have finished, used to elect the one
    // that writes the header last.
    uint32_t arrivals;
};

// Encode one block and place it in the archive: stage its encoded words,
// reserve their home out of the batch member's running tail, move them there,
// and record the block's entry in the header.
//
// Shared by the multi-launch encode kernel and the cooperative single-launch
// one so that the two cannot drift apart -- they emit the same archive format
// and must keep agreeing on it.
template <typename Symbols, int ProbBits, int BlockSize>
__device__ void ansEncodeAndPlaceBlock(
    // lane id within the warp doing this block
    int laneId,
    // this block's index within its batch member
    uint32_t block, uint32_t numBlocks,
    // number of symbols in the whole batch member
    uint32_t uncompressedWords,
    // where this member's symbols come from
    const void* inBase,
    // archive being written
    ANSCoalescedHeader* __restrict__ headerOut,
    // staging for the encoded words, already advanced to this batch member
    uint8_t* __restrict__ staging, uint32_t stagingBlockStride,
    // this batch member's running tail of reserved compressed words
    uint32_t* __restrict__ tailWords,
    // the encoding table, in smem
    const uint4* __restrict__ smemLookup) {
    // How big is this block? Only the final one can be short, and the warp that
    // owns it encodes it here -- there is no separate kernel for the partial
    // one.
    uint32_t start = block * BlockSize;
    uint32_t end = min(start + BlockSize, uncompressedWords);
    auto blockSize = end - start;

    constexpr uint32_t kAlignWords = kBlockAlignment / sizeof(ANSEncodedT);

    // Where this block's encoded words are staged, and where its final ANS
    // state goes: straight into the header, rather than into the staging slot
    // for a later pass to move.
    auto stagedWords =
        (ANSEncodedT*)(staging + (size_t)block * stagingBlockStride);
    auto stateOut = headerOut->getWarpStates() + block;

    auto syms = Symbols::make(inBase, start);

    uint32_t outWords;
    if (blockSize == BlockSize) {
        outWords = ANSEncodeWarpFullBlock<ProbBits, BlockSize, Symbols>::encode(
            laneId, syms, smemLookup, stateOut, stagedWords);
    } else {
        outWords = ansEncodeWarpBlock<ProbBits, Symbols>(
            laneId, syms, blockSize, smemLookup, stateOut, stagedWords);
    }

    auto alignedWords = roundUp(outWords, kAlignWords);

    // If the bound on max compressed size is not correct, this assert will go
    // off. This block of data was then somewhat adversarial in terms of
    // incompressibility. In this case, the getRawCompBlockMaxSize max estimate
    // needs to increase.
    assert(alignedWords * sizeof(ANSEncodedT) <=
           getMaxBlockSizeCoalesced(BlockSize));

    // Reserve this block's home out of the batch member's running tail
    uint32_t destWords = 0;

    if (laneId == 0) {
        destWords = atomicAdd(tailWords, alignedWords);

        // Per-block information:
        // (uint16: uncompressedWords, uint16: compressedWords)
        // uint32: blockCompressedWordStart
        headerOut->getBlockWords(numBlocks)[block] =
            uint2{(blockSize << 16) | outWords, destWords};
    }

    destWords = __shfl_sync(0xffffffff, destWords, 0);

    // Move the staged words into the reservation. Both ends are kBlockAlignment
    // aligned -- the staging slot by its stride, the reservation because every
    // block reserves a whole multiple of kAlignWords -- so this is a uint4
    // copy.
    auto destOut =
        (uint4*)(headerOut->getBlockDataStart(numBlocks) + destWords);
    auto stagedOut = (const uint4*)stagedWords;

    for (uint32_t i = laneId; i < alignedWords / kAlignWords; i += kWarpSize) {
        destOut[i] = stagedOut[i];
    }
}

// Write the archive header, its symbol probabilities and the reported size.
//
// Only valid once every block of this batch member has reserved its home, since
// the header carries the total compressed length.
template <int Threads>
__device__ void ansEncodeWriteArchive(
    uint32_t tid, ANSCoalescedHeader* __restrict__ headerOut,
    uint32_t numBlocks, uint32_t uncompressedWords,
    uint32_t totalCompressedWords, const uint4* __restrict__ smemLookup,
    uint32_t probBits, bool useChecksum, uint32_t checksum,
    uint32_t* __restrict__ compressedBytesOut) {
    if (tid == 0) {
        // Value-initialized so the header's reserved words are zero rather than
        // whatever this stack held; archives are meant to be persistable.
        ANSCoalescedHeader header{};
        header.setMagicAndVersion();
        header.setNumBlocks(numBlocks);
        header.setTotalUncompressedWords(uncompressedWords);
        header.setTotalCompressedWords(totalCompressedWords);
        header.setProbBits(probBits);
        header.setUseChecksum(useChecksum);

        if (useChecksum) {
            header.setChecksum(checksum);
        }

        if (compressedBytesOut) {
            *compressedBytesOut = header.getTotalCompressedSize();
        }

        *headerOut = header;
    }

    // Write out pdf, from the copy of the table this CTA already has
    auto probsOut = headerOut->getSymbolProbs();

    for (int i = tid; i < kNumSymbols; i += Threads) {
        probsOut[i] = smemLookup[i].x;
    }

    // The block words array is rounded up to a whole uint2 per block so that
    // the data region following it stays kBlockAlignment aligned, so with an
    // odd block count its final slot is padding. Zero it: that slot lies inside
    // the reported archive size, and leaving it holding whatever the output
    // buffer previously contained would both put uninitialized memory into an
    // archive that is meant to be persistable and make two archives of the same
    // input differ. The coalescing pass this replaced never wrote it either.
    constexpr uint32_t kBlockWordsAlign = kBlockAlignment / sizeof(uint2);
    static_assert(kBlockWordsAlign > 0, "");

    auto blockWordsOut = headerOut->getBlockWords(numBlocks);

    for (uint32_t i = numBlocks + tid; i < roundUp(numBlocks, kBlockWordsAlign);
         i += Threads) {
        blockWordsOut[i] = uint2{0, 0};
    }
}

// Encode every block of a batch member, place each block's output in the
// archive, and write the archive header.
//
// A block's encoded words cannot go straight to their final home: the length of
// a block is only known once that block has been encoded, while its home
// depends on the lengths of blocks other warps are encoding concurrently. So
// the words are staged in a worst-case-sized slot, the warp then reserves
// exactly its own aligned length out of the batch member's running tail and
// copies the staged words into the reservation while they are still hot in
// L1/L2. Reserving from a shared tail means blocks land in completion order
// rather than block order; that is legal because the archive records every
// block's start offset in its blockWords entry and the decoder looks blocks up
// by that offset instead of assuming they are laid out in order. What it costs
// is that an archive's byte layout is no longer reproducible run to run (it
// never was bit-reproducible anyway: the header's unused words were left
// uninitialized). What it buys is the removal of the prefix sum and of a
// coalescing pass that read and wrote every encoded word a second time.
template <typename InProvider, typename OutProvider, typename Symbols,
          int Threads, int ProbBits, int BlockSize>
__global__ void ansEncodeBatchFused(
    // Input data for all blocks
    InProvider inProvider,
    // Archive each batch member is written into
    OutProvider outProvider,
    // maximum number of blocks across all the batch
    uint32_t maxNumCompressedBlocks,
    // stride in bytes between each block's staging slot
    uint32_t stagingBlockStride,
    // per-block staging for the encoded words
    // [batch][maxNumCompressedBlocks][stagingBlockStride]
    uint8_t* __restrict__ staging,
    // per-batch running tail and CTA arrival counters, zeroed on entry
    ANSEncodeCounters* __restrict__ counters,
    // the encoding table that we will load into smem
    // [batch][kNumSymbols]
    const uint4* __restrict__ table,
    // optional per-batch checksum of the uncompressed input, only read if
    // useChecksum
    const uint32_t* __restrict__ checksum, uint32_t probBits, bool useChecksum,
    // optional per-batch size in bytes of the produced archive
    uint32_t* __restrict__ compressedBytes) {
    // which batch element we are processing
    auto batch = blockIdx.y;
    auto tid = threadIdx.x;

    // Size of this problem, in ANSDecodedT words, and the blocks it splits into
    uint32_t uncompressedWords = inProvider.getBatchSize(batch);
    uint32_t numBlocks = divUp(uncompressedWords, BlockSize);

    auto headerOut = (ANSCoalescedHeader*)outProvider.getBatchStart(batch);

    __shared__ uint4 smemLookup[kNumSymbols];

    // we always have at least 256 threads
    if (tid < kNumSymbols) {
        smemLookup[tid] = table[batch * kNumSymbols + tid];
    }

    __syncthreads();

    // so we know the block is warp uniform
    int block =
        __shfl_sync(0xffffffff, (blockIdx.x * Threads + tid) / kWarpSize, 0);
    int laneId = getLaneId();

    if (block < numBlocks) {
        ansEncodeAndPlaceBlock<Symbols, ProbBits, BlockSize>(
            laneId, block, numBlocks, uncompressedWords,
            inProvider.getBatchStart(batch), headerOut,
            staging +
                batch * (size_t)maxNumCompressedBlocks * stagingBlockStride,
            stagingBlockStride, &counters[batch].tailWords, smemLookup);
    }

    // Elect the last CTA of this batch member to write the header, which needs
    // the total compressed length and so cannot be written until every block of
    // that member has reserved its own.
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
        ansEncodeWriteArchive<Threads>(
            tid, headerOut, numBlocks, uncompressedWords,
            counters[batch].tailWords, smemLookup, probBits, useChecksum,
            useChecksum ? checksum[batch] : 0,
            compressedBytes ? compressedBytes + batch : nullptr);
    }
}

// How many CTAs per SM the cooperative compressors ask the compiler to leave
// room for. The second argument of __launch_bounds__ caps registers per thread
// at 65536 / (Threads * this), and without it the fused kernels come out at
// 77-83 registers -- which on a 65536 register per SM device permits only 3
// CTAs of 256 threads, against 4 for the multi-launch encode kernel at 54
// registers. The ANS encode loop is bound by the serial dependency in its state
// update rather than by issue rate, so resident warps are precisely the
// resource that hides it; a fused kernel that starts out with fewer of them
// than the kernels it replaces loses on large inputs whatever it saves on
// launches.
//
// 6 is reachable: with it the fused kernels come out at 40 registers and no
// spilling (LOCAL:0), which together with the overlaid per-phase scratch below
// (23888 -> 15617 bytes) gives 6 CTAs of 256 threads per SM -- 100% occupancy,
// and more than the 4 the multi-launch encode kernel gets. Measured effect on
// co-resident blocks on a 128 SM device: 384 -> 768.
constexpr int kCoopMinBlocksPerSM = 6;

// Single-launch compression: symbol statistics, probability normalization,
// per-block encode and archive finalization in one cooperative kernel.
//
// The two grid-wide dependencies compression has are "every CTA's histogram
// contribution is visible before anyone normalizes" and "every block has
// reserved its home before the total goes into the header". Both need a real
// grid barrier, which is what cudaLaunchCooperativeKernel plus
// cooperative_groups::grid_group::sync() provides -- and what a plain launch
// cannot, since its CTAs come and go in waves and a spinning software barrier
// would deadlock as soon as the grid exceeded what the device can keep
// resident.
//
// What that costs, and why this is opt-in rather than the default:
//   - the two grid barriers are pure overhead the multi-launch path does not
//   pay,
//     because there kernel boundaries are the barriers. Measured at ~2.0 us
//     each on a 768 CTA grid, against a total of ~55 us to compress 32 MiB;
//   - the grid is capped at the device's resident block count, so the encode
//     phase has to be a grid-stride loop over blocks instead of one warp per
//     block. Occupancy itself is not a cost any more: the phases' shared memory
//     is overlaid (grid.sync() separates them, so at most one phase's scratch
//     is live) and __launch_bounds__ caps registers without spilling, which
//     together reach 6 CTAs of 256 threads per SM -- more than the 4 the
//     multi-launch encode kernel gets;
//   - while it runs, it occupies the whole device, so nothing else can be
//     overlapped with it on another stream.
// Where it wins is where those costs do not bite: small messages, where the
// multi-launch path is host-submit bound and the GPU is idle waiting for it.
// Measured, one batch member: 11-14% faster up to 16 MiB, ~8% slower at 64 MiB.
//
// The float codec has its own counterpart (floatCompressCooperative) rather
// than sharing this one, because its first phase is splitFloat -- extract the
// compressible byte, write the raw remainder into the archive, histogram -- and
// not a plain histogram. The two share everything after that phase.
template <typename InProvider, typename OutProvider, typename Symbols,
          int Threads, int ProbBits, int BlockSize>
__global__
__launch_bounds__(Threads, kCoopMinBlocksPerSM) void ansEncodeBatchCooperative(
    // Input data for all blocks
    InProvider inProvider,
    // Archive each batch member is written into
    OutProvider outProvider,
    // maximum number of blocks across all the batch
    uint32_t maxNumCompressedBlocks,
    // stride in bytes between each block's staging slot
    uint32_t stagingBlockStride,
    // per-block staging for the encoded words
    uint8_t* __restrict__ staging,
    // per-batch running tail and CTA arrival counters, zeroed on entry
    ANSEncodeCounters* __restrict__ counters,
    // per-batch symbol histogram, zeroed on entry
    uint32_t* __restrict__ histogram,
    // optional per-batch checksum of the uncompressed input, zeroed on entry
    // and only accumulated / read if useChecksum
    uint32_t* __restrict__ checksum, uint32_t probBits, bool useChecksum,
    // optional per-batch size in bytes of the produced archive
    uint32_t* __restrict__ compressedBytes) {
    auto grid = cooperative_groups::this_grid();

    // which batch element we are processing
    auto batch = blockIdx.y;
    auto tid = threadIdx.x;

    uint32_t uncompressedWords = inProvider.getBatchSize(batch);
    uint32_t numBlocks = divUp(uncompressedWords, BlockSize);

    auto inBase = inProvider.getBatchStart(batch);
    auto headerOut = (ANSCoalescedHeader*)outProvider.getBatchStart(batch);
    auto histogramOut = histogram + batch * kNumSymbols;

    // Scratch for every phase, overlaid. grid.sync() separates the phases, so
    // at most one of these is live at a time -- but shared memory declared
    // inside the functions each phase used to be its own kernel is additive,
    // and the sum measured 23888 bytes. On a 100 KiB per SM device that is 4
    // CTAs, and this kernel needs to beat that to justify fusing: the ANS
    // encode loop is bound by the serial dependency in its state update, so
    // resident warps are exactly what hides it. Overlaying is also what makes
    // kCoopMinBlocksPerSM reachable.
    //
    // The histogram buckets are dead by the time the table is built: phase 1
    // reduces them into the gmem histogram before the barrier, and phase 2
    // reads that, not the buckets.
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

    // 1. Symbol statistics over this batch member, and optionally its checksum.
    // Both are grid-stride reductions over the same input, so doing them in one
    // phase reads the input once instead of once per reduction.
    histogramSingle<Threads>((const ANSDecodedT*)inBase, uncompressedWords,
                             histogramOut, scratch.stats.histogram);

    if (useChecksum) {
        checksumSingle<Threads>((const uint8_t*)inBase, uncompressedWords,
                                checksum + batch, scratch.stats.reduce);
    }

    // Nothing may normalize until every CTA's histogram contribution has landed
    grid.sync();

    // 2. Normalize this member's histogram into this CTA's own smem table.
    // Every CTA of the member redoes it, which is what lets the table stay in
    // smem -- there is no gmem table buffer and no third barrier to publish one
    // through.
    auto& smemLookup = scratch.encode.lookup;

    normalizeProbabilitiesFromHistogram<Threads>(
        histogramOut, uncompressedWords, probBits, smemLookup,
        scratch.encode.normalize);

    // The normalizer's last act is each thread writing its own table entry, and
    // the encode loop below indexes the table by symbol, so the writes have to
    // be visible across the whole CTA first.
    __syncthreads();

    // 3. Encode every block of this member and place it in the archive. The
    // grid is capped by co-residency, so a warp loops over blocks rather than
    // owning one -- the decode kernel has always worked this way.
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
            laneId, block, numBlocks, uncompressedWords, inBase, headerOut,
            stagingBatch, stagingBlockStride, &counters[batch].tailWords,
            smemLookup);
    }

    // 4. Elect the last CTA of this batch member to write the header, which
    // needs the total compressed length and so cannot be written until every
    // block of that member has reserved its own. This is a plain arrival
    // counter rather than a grid barrier: the header does not have to be
    // written before the kernel ends, only after this member's blocks are all
    // placed.
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
        ansEncodeWriteArchive<Threads>(
            tid, headerOut, numBlocks, uncompressedWords,
            counters[batch].tailWords, smemLookup, probBits, useChecksum,
            useChecksum ? checksum[batch] : 0,
            compressedBytes ? compressedBytes + batch : nullptr);
    }
}

// Symbols defaults to reading the input as the raw bytes it is; the float codec
// passes its own reader so that the compressible bytes are re-extracted from
// the float words during encoding instead of being staged in a temporary
// buffer.
template <typename InProvider, typename OutProvider,
          typename Symbols = ANSSymbolBytes>
void ansEncodeBatchDevice(StackDeviceMemory& res, const ANSCodecConfig& config,
                          uint32_t numInBatch, InProvider inProvider,
                          const uint32_t* histogram_dev, uint32_t maxSize,
                          OutProvider outProvider, uint32_t* outSize_dev,
                          cudaStream_t stream) {
    auto maxUncompressedWords = maxSize / sizeof(ANSDecodedT);
    uint32_t maxNumCompressedBlocks =
        divUp(maxUncompressedWords, kDefaultBlockSize);

    // Only kANSDefaultProbBits is instantiated. Every extra precision
    // multiplies each kernel here by the number of provider combinations, and
    // that matrix -- not any single kernel -- is what dominates build time.
    // Validate before anything is allocated so an empty batch still reports it.
    CHECK_EQ(config.probBits, kANSDefaultProbBits)
        << "this build of dietgpu only instantiates pdf precision "
        << kANSDefaultProbBits << "; see ANSCodecConfig::probBits";

    // Single-launch path, when asked for and when this request can be served
    // cooperatively. Falling back rather than failing keeps the option usable
    // at any batch size: a cooperative launch is rejected outright if its grid
    // cannot be co-resident, and having more batch members than that is
    // perfectly legitimate for the multi-launch path below. A caller-supplied
    // histogram also rules it out, since phase 1 computes its own.
    if constexpr (std::is_same<Symbols, ANSSymbolBytes>::value) {
        if (config.useSingleKernel && !histogram_dev && numInBatch > 0) {
            // histogramSingle asserts Threads == kNumSymbols
            constexpr int kThreads = kNumSymbols;

            auto kernel =
                ansEncodeBatchCooperative<InProvider, OutProvider, Symbols,
                                          kThreads, kANSDefaultProbBits,
                                          kDefaultBlockSize>;

            // A cooperative launch is rejected outright, not degraded, if the
            // device cannot keep the whole grid resident or does not support
            // cooperative launch at all. Treat both as "take the multi-launch
            // path" rather than letting CUDA_VERIFY abort a request this
            // library can otherwise serve perfectly well.
            uint32_t maxGrid = getCooperativeLaunchSupportCurrentDevice()
                                   ? getMaxResidentBlocks(kernel, kThreads)
                                   : 0;

            if (numInBatch <= maxGrid) {
                uint32_t stagingBlockStride =
                    getMaxBlockSizeCoalesced(kDefaultBlockSize);

                auto staging_dev = res.alloc<uint8_t>(
                    stream, (size_t)numInBatch * maxNumCompressedBlocks *
                                stagingBlockStride);

                // Histogram, checksum and counters in one reservation so that a
                // single memset zeroes all three; they all accumulate.
                uint32_t scratchWords = numInBatch * (kNumSymbols + 1 + 2);
                auto scratch_dev = res.alloc<uint32_t>(stream, scratchWords);

                CUDA_VERIFY(cudaMemsetAsync(scratch_dev.data(), 0,
                                            sizeof(uint32_t) * scratchWords,
                                            stream));

                uint32_t* histogramArg = scratch_dev.data();
                uint32_t* checksumArg = histogramArg + numInBatch * kNumSymbols;
                ANSEncodeCounters* countersArg =
                    (ANSEncodeCounters*)(checksumArg + numInBatch);
                uint8_t* stagingArg = staging_dev.data();
                uint32_t probBitsArg = config.probBits;
                bool useChecksumArg = config.useChecksum;

                // The encode phase is a grid-stride loop, so the grid only has
                // to be big enough to keep the device busy -- not one CTA per
                // block, which would blow through the co-residency cap on
                // anything but a small input. At least one CTA per batch member
                // even when there are no blocks: that CTA is what writes the
                // member's header, and gridDim.x == 0 is an invalid launch
                // configuration rather than a no-op.
                uint32_t gridX =
                    std::max(std::min(divUp((int)maxNumCompressedBlocks,
                                            kThreads / kWarpSize),
                                      (int)std::max(maxGrid / numInBatch, 1U)),
                             1);
                auto grid = dim3(gridX, numInBatch);

                void* args[] = {(void*)&inProvider,
                                (void*)&outProvider,
                                (void*)&maxNumCompressedBlocks,
                                (void*)&stagingBlockStride,
                                (void*)&stagingArg,
                                (void*)&countersArg,
                                (void*)&histogramArg,
                                (void*)&checksumArg,
                                (void*)&probBitsArg,
                                (void*)&useChecksumArg,
                                (void*)&outSize_dev};

                CUDA_VERIFY(cudaLaunchCooperativeKernel(
                    (void*)kernel, grid, dim3(kThreads), args, 0, stream));

                CUDA_TEST_ERROR();

                return;
            }

            // The flag is opt-in, so ignoring it silently would leave the
            // caller believing compression ran in a single launch. Warn once
            // rather than on every call; `warned` is per template
            // instantiation, so at most a handful of provider combinations can
            // each warn once.
            static std::atomic<bool> warned{false};

            if (!warned.exchange(true)) {
                if (maxGrid == 0) {
                    LOG(WARNING)
                        << "ANSCodecConfig::useSingleKernel is set, but this "
                           "device reports no cooperative launch support; "
                           "compression will keep using the multi-launch path";
                } else {
                    LOG(WARNING)
                        << "ANSCodecConfig::useSingleKernel is set, but a "
                           "batch "
                           "of "
                        << numInBatch << " members exceeds the " << maxGrid
                        << " co-resident blocks this device allows for a "
                           "cooperative launch; compression will keep using "
                           "the "
                           "multi-launch path";
                }
            }
        }
    }

    // 1. Compute symbol statistics
    auto table_dev = res.alloc<uint4>(stream, numInBatch * kNumSymbols);

    if (histogram_dev) {
        // use pre-calculated histogram
        ansCalcWeights(numInBatch, config.probBits, inProvider, histogram_dev,
                       table_dev.data(), stream);
    } else {
        auto tempHistogram_dev =
            res.alloc<uint32_t>(stream, numInBatch * kNumSymbols);

        // need to calculate a histogram
        ansHistogramBatch(numInBatch, inProvider, tempHistogram_dev.data(),
                          stream);

        ansCalcWeights(numInBatch, config.probBits, inProvider,
                       tempHistogram_dev.data(), table_dev.data(), stream);
    }

    // 2. Compute checksum on input data (optional)
    auto checksum_dev = res.alloc<uint32_t>(stream, numInBatch);
    if (config.useChecksum) {
        checksumBatch(numInBatch, inProvider, checksum_dev.data(), stream);
    }

    // 3. Staging for the per-block encoded words, plus the per-batch counters
    // the fused kernel reserves each block's home from.
    //
    // This staging is the one part of compression's temporary memory that
    // cannot be removed without changing the archive format: a block's encoded
    // words have to end up contiguous, how many there are is not known until
    // the block has been encoded, and holding a full block's worst case (5 KiB)
    // in smem or in registers instead would cost far more occupancy than the
    // resulting copy saves. What did go away is the 128 bytes per block of ANS
    // warp state that used to be staged here for a later pass to relocate --
    // the encoding warp writes that straight into the header now -- along with
    // the two numBlocks sized arrays of per-block lengths and their prefix sum.
    uint32_t stagingBlockStride = getMaxBlockSizeCoalesced(kDefaultBlockSize);

    auto staging_dev =
        res.alloc<uint8_t>(stream, (size_t)numInBatch * maxNumCompressedBlocks *
                                       stagingBlockStride);

    auto counters_dev = res.alloc<ANSEncodeCounters>(stream, numInBatch);

    // Both counters accumulate, so they have to start from zero
    CUDA_VERIFY(cudaMemsetAsync(counters_dev.data(), 0,
                                sizeof(ANSEncodeCounters) * numInBatch,
                                stream));

    // 4. Encode every block, place it in the archive and write the header.
    // Even if there is nothing to compress we still need to create a
    // compression header, so there is always at least one CTA per batch member.
    {
        constexpr int kThreads = 256;

        auto grid = dim3(
            std::max(divUp((int)maxNumCompressedBlocks, kThreads / kWarpSize),
                     1),
            numInBatch);

        ansEncodeBatchFused<InProvider, OutProvider, Symbols, kThreads,
                            kANSDefaultProbBits, kDefaultBlockSize>
            <<<grid, kThreads, 0, stream>>>(
                inProvider, outProvider, maxNumCompressedBlocks,
                stagingBlockStride, staging_dev.data(), counters_dev.data(),
                table_dev.data(), checksum_dev.data(), config.probBits,
                config.useChecksum, outSize_dev);
    }

    CUDA_TEST_ERROR();
}

}  // namespace dietgpu
