/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/float/GpuFloatUtils.cuh"
#include "dietgpu/utils/StackDeviceMemory.h"

using namespace dietgpu;

uint16_t float32ToBFloat16(float f) {
    // FIXME: does not round to nearest even
    static_assert(sizeof(float) == sizeof(uint32_t), "");
    uint32_t x;
    std::memcpy(&x, &f, sizeof(float));

    x >>= 16;
    return x;
}

uint16_t float32ToFloat16(float f) {
    static_assert(sizeof(float) == sizeof(uint32_t), "");
    uint32_t x;
    std::memcpy(&x, &f, sizeof(float));

    uint32_t u = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
    uint32_t sign, exponent, mantissa;

    // Get rid of +NaN/-NaN case first.
    if (u > 0x7f800000U) {
        return 0x7fffU;
    }

    sign = ((x >> 16) & 0x8000);

    // Get rid of +Inf/-Inf, +0/-0.
    if (u > 0x477fefffU) {
        return sign | 0x7c00U;
    }
    if (u < 0x33000001U) {
        return (sign | 0x0000);
    }

    exponent = ((u >> 23) & 0xff);
    mantissa = (u & 0x7fffff);

    if (exponent > 0x70) {
        shift = 13;
        exponent -= 0x70;
    } else {
        shift = 0x7e - exponent;
        exponent = 0;
        mantissa |= 0x800000;
    }
    lsb = (1 << shift);
    lsb_s1 = (lsb >> 1);
    lsb_m1 = (lsb - 1);

    // Round to nearest even.
    remainder = (mantissa & lsb_m1);
    mantissa >>= shift;
    if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
        ++mantissa;
        if (!(mantissa & 0x3ff)) {
            ++exponent;
            mantissa = 0;
        }
    }

    return (sign | (exponent << 10) | mantissa);
}

template <FloatType FT>
struct GenerateFloat;

template <>
struct GenerateFloat<FloatType::kFloat16> {
    static FloatTypeInfo<FloatType::kFloat16>::WordT gen(float v) {
        return float32ToFloat16(v);
    }
};

template <>
struct GenerateFloat<FloatType::kBFloat16> {
    static FloatTypeInfo<FloatType::kBFloat16>::WordT gen(float v) {
        return float32ToBFloat16(v);
    }
};

template <>
struct GenerateFloat<FloatType::kFloat32> {
    static FloatTypeInfo<FloatType::kFloat32>::WordT gen(float v) {
        FloatTypeInfo<FloatType::kFloat32>::WordT out;
        std::memcpy(&out, &v, sizeof(float));
        return out;
    }
};

template <FloatType FT>
std::vector<typename FloatTypeInfo<FT>::WordT> generateFloats(int num) {
    std::mt19937 gen(10 + num);
    std::normal_distribution<float> dist;

    auto out = std::vector<typename FloatTypeInfo<FT>::WordT>(num);
    for (auto& v : out) {
        v = GenerateFloat<FT>::gen(dist(gen));
    }

    return out;
}

template <FloatType FT>
std::vector<uint32_t> runBatchPointerTest(
    StackDeviceMemory& res, int probBits,
    const std::vector<uint32_t>& batchSizes, bool singleKernel = false,
    // Shift every decompression output pointer by this many float words. Used
    // to exercise outputs that are float word aligned but not 16 byte aligned.
    uint32_t outOffsetElems = 0) {
    using FTI = FloatTypeInfo<FT>;

    // run on a different stream to test stream assignment
    auto stream = CudaStream::makeNonBlocking();

    int numInBatch = batchSizes.size();
    uint32_t totalSize = 0;
    uint32_t maxSize = 0;
    for (auto v : batchSizes) {
        totalSize += v;
        maxSize = std::max(maxSize, v);
    }

    auto maxCompressedSize = getMaxFloatCompressedSize(FT, maxSize);

    auto orig = generateFloats<FT>(totalSize);
    auto orig_dev = res.copyAlloc(stream, orig);

    auto inPtrs = std::vector<const void*>(batchSizes.size());
    {
        uint32_t curOffset = 0;
        for (int i = 0; i < inPtrs.size(); ++i) {
            inPtrs[i] = (const typename FTI::WordT*)orig_dev.data() + curOffset;
            curOffset += batchSizes[i];
        }
    }

    auto enc_dev = res.alloc<uint8_t>(stream, numInBatch * maxCompressedSize);

    auto encPtrs = std::vector<void*>(batchSizes.size());
    {
        for (int i = 0; i < inPtrs.size(); ++i) {
            encPtrs[i] = (uint8_t*)enc_dev.data() + i * maxCompressedSize;
        }
    }

    auto outBatchSize_dev = res.alloc<uint32_t>(stream, numInBatch);

    auto compConfig = FloatCompressConfig(
        FT, ANSCodecConfig(probBits, false, singleKernel), false, true);

    floatCompress(res, compConfig, numInBatch, inPtrs.data(), batchSizes.data(),
                  encPtrs.data(), outBatchSize_dev.data(), stream);

    auto encSize = outBatchSize_dev.copyToHost(stream);
    for (auto v : encSize) {
        // Reported compressed sizes in bytes should be a multiple of 16 for
        // aligned packing
        EXPECT_EQ(v % 16, 0);
    }

    // Decode data. The extra outOffsetElems words at the front are padding so
    // that the outputs can be deliberately misaligned.
    auto dec_dev =
        res.alloc<typename FTI::WordT>(stream, totalSize + outOffsetElems);

    auto decPtrs = std::vector<void*>(batchSizes.size());
    {
        uint32_t curOffset = outOffsetElems;
        for (int i = 0; i < inPtrs.size(); ++i) {
            decPtrs[i] = (typename FTI::WordT*)dec_dev.data() + curOffset;
            curOffset += batchSizes[i];
        }
    }

    auto outSuccess_dev = res.alloc<uint8_t>(stream, numInBatch);
    auto outSize_dev = res.alloc<uint32_t>(stream, numInBatch);

    auto decompConfig =
        FloatDecompressConfig(FT, ANSCodecConfig(probBits), false, true);

    floatDecompress(res, decompConfig, numInBatch, (const void**)encPtrs.data(),
                    decPtrs.data(), batchSizes.data(), outSuccess_dev.data(),
                    outSize_dev.data(), stream);

    auto outSuccess = outSuccess_dev.copyToHost(stream);
    auto outSize = outSize_dev.copyToHost(stream);

    for (int i = 0; i < outSuccess.size(); ++i) {
        EXPECT_TRUE(outSuccess[i]);
        EXPECT_EQ(outSize[i], batchSizes[i]);
    }

    auto dec = dec_dev.copyToHost(stream);

    // Drop the misalignment padding so that what follows compares the decoded
    // payload only
    dec.erase(dec.begin(), dec.begin() + outOffsetElems);

    for (int i = 0; i < orig.size(); ++i) {
        if (orig[i] != dec[i]) {
            printf("mismatch at %d / %d: 0x%08X 0x%08X\n", i, (int)orig.size(),
                   orig[i], dec[i]);
            break;
        }
    }

    EXPECT_EQ(orig, dec);

    return encSize;
}

std::vector<uint32_t> runBatchPointerTest(
    StackDeviceMemory& res, FloatType ft, int probBits,
    const std::vector<uint32_t>& batchSizes, bool singleKernel = false,
    uint32_t outOffsetElems = 0) {
    switch (ft) {
        case FloatType::kFloat16:
            return runBatchPointerTest<FloatType::kFloat16>(
                res, probBits, batchSizes, singleKernel, outOffsetElems);
        case FloatType::kBFloat16:
            return runBatchPointerTest<FloatType::kBFloat16>(
                res, probBits, batchSizes, singleKernel, outOffsetElems);
        case FloatType::kFloat32:
            return runBatchPointerTest<FloatType::kFloat32>(
                res, probBits, batchSizes, singleKernel, outOffsetElems);
        default:
            CHECK(false);
            return {};
    }
}

void runBatchPointerTest(StackDeviceMemory& res, FloatType ft, int probBits,
                         int numInBatch, uint32_t multipleOf = 1) {
    std::mt19937 gen(10 + numInBatch);
    std::uniform_int_distribution<uint32_t> dist(1, 10000);

    auto batchSizes = std::vector<uint32_t>(numInBatch);
    for (auto& v : batchSizes) {
        v = roundUp(dist(gen), multipleOf);
    }

    runBatchPointerTest(res, ft, probBits, batchSizes);
}

TEST(FloatTest, Batch) {
    auto res = makeStackMemory();

    // This build only instantiates kANSDefaultProbBits; see
    // ANSCodecConfig::probBits for why precision 9 is no longer exercised.
    const int probBits = kANSDefaultProbBits;
    for (auto ft :
         {FloatType::kFloat16, FloatType::kBFloat16, FloatType::kFloat32}) {
        for (auto numInBatch : {1, 3, 16, 23}) {
            runBatchPointerTest(res, ft, probBits, numInBatch);
            // Also test the case where there is uniform 16 byte alignment
            // across all batches
            runBatchPointerTest(res, ft, probBits, numInBatch, 16);
        }
    }
}

TEST(FloatTest, LargeBatch) {
    auto res = makeStackMemory();

    auto batchSizes = std::vector<uint32_t>(256);
    for (auto& v : batchSizes) {
        v = 512 * 1024;
    }

    for (auto ft :
         {FloatType::kFloat16, FloatType::kBFloat16, FloatType::kFloat32}) {
        runBatchPointerTest(res, ft, 10, batchSizes);
    }
}

TEST(FloatTest, BatchSize1) {
    auto res = makeStackMemory();

    const int probBits = kANSDefaultProbBits;
    for (auto ft :
         {FloatType::kFloat16, FloatType::kBFloat16, FloatType::kFloat32}) {
        runBatchPointerTest(res, ft, probBits, {1});
        runBatchPointerTest(res, ft, probBits, {13, 1});
        runBatchPointerTest(res, ft, probBits, {12345, 1, 8083, 1, 17});
    }
}

TEST(FloatTest, SingleKernel) {
    auto res = makeStackMemory();

    const int probBits = kANSDefaultProbBits;

    // All three float types, because the single-launch kernel routes them
    // differently: float16 and bfloat16 re-extract the compressible byte from
    // the input while encoding, whereas float32 still stages an exponent array
    // and so takes the raw-bytes symbol reader. The shapes cover a zero sized
    // member (header only, no blocks, and the float header's size field feeds
    // the archive offset), a sub-block member, an exact multiple of the 4 KiB
    // block, and a mixed batch.
    //
    // The grid-stride encode loop itself is covered by ANSTest.SingleKernel,
    // which reaches it at a fraction of the memory cost: both kernels drive the
    // same ansEncodeAndPlaceBlock over the same loop shape.
    std::vector<std::vector<uint32_t>> batchSets = {
        {0},
        {1},
        {4095},
        {4096},
        {4097},
        {1000, 4096, 100000, 1},
        {1 << 20, 1 << 20},
    };

    // More members than any device can keep co-resident, so the cooperative
    // launch would be rejected: this has to fall back to the multi-launch path
    // and still be correct, not fail.
    batchSets.push_back(std::vector<uint32_t>(1024, 512));

    for (auto ft :
         {FloatType::kFloat16, FloatType::kBFloat16, FloatType::kFloat32}) {
        for (auto& sizes : batchSets) {
            auto single = runBatchPointerTest(res, ft, probBits, sizes, true);
            auto multi = runBatchPointerTest(res, ft, probBits, sizes, false);

            // The two paths must agree on every member's compressed size. Their
            // byte layouts need not match -- each block reserves its home from
            // a shared tail -- but the histogram is an exact integer sum, the
            // normalization is the same function at the same thread count, and
            // the split is the same one, so the per-block lengths and therefore
            // the totals are identical.
            EXPECT_EQ(single, multi);
        }
    }
}

TEST(FloatTest, UnalignedOutput) {
    auto res = makeStackMemory();

    const int probBits = kANSDefaultProbBits;

    // Decompression is a single pass at any output alignment: the writer joins
    // each decoded symbol with its stored remainder and stores one float word
    // at a time, so the word alignment the API already requires is all it
    // needs. An output pointer 1, 2 or 3 words into an aligned allocation sits
    // at 2/4/6 bytes (float16, bfloat16) or 4/8/12 bytes (float32) past it --
    // never 16 byte aligned. That is precisely the case that used to be routed
    // to a second kernel through a temporary the size of the whole
    // decompressed output.
    //
    // The sizes straddle the 4 KiB ANS block, so the per-block output offsets
    // within a member cover a mix of alignments as well, and the empty member
    // checks that a member with nothing to write writes nothing.
    std::vector<std::vector<uint32_t>> batchSets = {
        {0},
        {1},
        {4095},
        {4096},
        {4097},
        {1, 2, 3, 4095, 4096, 4097},
        {12345, 1, 8083, 1, 17},
    };

    for (auto ft :
         {FloatType::kFloat16, FloatType::kBFloat16, FloatType::kFloat32}) {
        for (auto& sizes : batchSets) {
            for (auto offset : {1u, 2u, 3u}) {
                runBatchPointerTest(res, ft, probBits, sizes, false, offset);
            }
        }
    }
}
