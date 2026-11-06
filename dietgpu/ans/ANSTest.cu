/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "dietgpu/ans/GpuANSCodec.h"
#include "dietgpu/utils/StackDeviceMemory.h"

using namespace dietgpu;

std::vector<uint8_t> generateSymbols(int num, float lambda = 20.0f) {
    std::random_device rd;
    std::mt19937 gen(10);
    std::exponential_distribution<float> dist(lambda);

    auto out = std::vector<uint8_t>(num);
    for (auto& v : out) {
        auto sample = std::min(dist(gen), 1.0f);

        v = sample * 256.0;
    }

    return out;
}

std::vector<GpuMemoryReservation<uint8_t>> toDevice(
    StackDeviceMemory& res, const std::vector<std::vector<uint8_t>>& vs,
    cudaStream_t stream) {
    auto out = std::vector<GpuMemoryReservation<uint8_t>>();

    for (auto& v : vs) {
        out.emplace_back(res.copyAlloc(stream, v, AllocType::Permanent));
    }

    return out;
}

std::vector<std::vector<uint8_t>> toHost(
    StackDeviceMemory& res,
    const std::vector<GpuMemoryReservation<uint8_t>>& vs, cudaStream_t stream) {
    auto out = std::vector<std::vector<uint8_t>>();

    for (auto& v : vs) {
        out.emplace_back(v.copyToHost(stream));
    }

    return out;
}

std::vector<GpuMemoryReservation<uint8_t>> buffersToDevice(
    StackDeviceMemory& res, const std::vector<uint32_t>& sizes,
    cudaStream_t stream) {
    auto out = std::vector<GpuMemoryReservation<uint8_t>>();

    for (auto& s : sizes) {
        out.emplace_back(res.alloc<uint8_t>(stream, s, AllocType::Permanent));
    }

    return out;
}

std::vector<std::vector<uint8_t>> genBatch(const std::vector<uint32_t>& sizes,
                                           double lambda) {
    auto out = std::vector<std::vector<uint8_t>>();

    for (auto s : sizes) {
        out.push_back(generateSymbols(s, lambda));
    }

    return out;
}

std::vector<uint32_t> runBatchPointer(StackDeviceMemory& res, int prec,
                                      const std::vector<uint32_t>& batchSizes,
                                      double lambda = 100.0,
                                      bool singleKernel = false) {
    // run on a different stream to test stream assignment
    auto stream = CudaStream::makeNonBlocking();

    int numInBatch = batchSizes.size();
    uint32_t maxSize = 0;
    for (auto v : batchSizes) {
        maxSize = std::max(maxSize, v);
    }

    auto outBatchStride = getMaxCompressedSize(maxSize);

    auto batch_host = genBatch(batchSizes, lambda);
    auto batch_dev = toDevice(res, batch_host, stream);

    auto inPtrs = std::vector<const void*>(batchSizes.size());
    {
        for (int i = 0; i < inPtrs.size(); ++i) {
            inPtrs[i] = batch_dev[i].data();
        }
    }

    auto enc_dev = res.alloc<uint8_t>(stream, numInBatch * outBatchStride);

    auto encPtrs = std::vector<void*>(batchSizes.size());
    for (int i = 0; i < inPtrs.size(); ++i) {
        encPtrs[i] = (uint8_t*)enc_dev.data() + i * outBatchStride;
    }

    auto outCompressedSize_dev = res.alloc<uint32_t>(stream, numInBatch);

    ansEncodeBatchPointer(res, ANSCodecConfig(prec, true, singleKernel),
                          numInBatch, inPtrs.data(), batchSizes.data(), nullptr,
                          encPtrs.data(), outCompressedSize_dev.data(), stream);

    auto encSize = outCompressedSize_dev.copyToHost(stream);
    for (auto v : encSize) {
        // Reported compressed sizes in bytes should be a multiple of 16 for
        // aligned packing
        EXPECT_EQ(v % 16, 0);
    }

    // Decode data
    auto dec_dev = buffersToDevice(res, batchSizes, stream);

    auto decPtrs = std::vector<void*>(batchSizes.size());
    for (int i = 0; i < inPtrs.size(); ++i) {
        decPtrs[i] = dec_dev[i].data();
    }

    auto outSuccess_dev = res.alloc<uint8_t>(stream, numInBatch);
    auto outSize_dev = res.alloc<uint32_t>(stream, numInBatch);

    ansDecodeBatchPointer(res, ANSCodecConfig(prec, true), numInBatch,
                          (const void**)encPtrs.data(), decPtrs.data(),
                          batchSizes.data(), outSuccess_dev.data(),
                          outSize_dev.data(), stream);

    auto outSuccess = outSuccess_dev.copyToHost(stream);
    auto outSize = outSize_dev.copyToHost(stream);

    for (int i = 0; i < outSuccess.size(); ++i) {
        EXPECT_TRUE(outSuccess[i]);
        EXPECT_EQ(outSize[i], batchSizes[i]);
    }

    auto dec_host = toHost(res, dec_dev, stream);
    EXPECT_EQ(batch_host, dec_host);

    return encSize;
}

void runBatchStride(StackDeviceMemory& res, int prec, int numInBatch,
                    int inBatchSize, double lambda = 100.0) {
    // run on a different stream to test stream assignment
    auto stream = CudaStream::makeNonBlocking();

    auto orig = generateSymbols(numInBatch * inBatchSize, lambda);
    auto orig_dev = res.copyAlloc(stream, orig);

    int outBatchStride = getMaxCompressedSize(inBatchSize);

    auto enc_dev = res.alloc<uint8_t>(stream, numInBatch * outBatchStride);

    auto outCompressedSize_dev = res.alloc<uint32_t>(stream, numInBatch);

    ansEncodeBatchStride(res, ANSCodecConfig(prec, true), numInBatch,
                         orig_dev.data(), inBatchSize, inBatchSize, nullptr,
                         enc_dev.data(), outBatchStride,
                         outCompressedSize_dev.data(), stream);

    auto encSize = outCompressedSize_dev.copyToHost(stream);
    for (auto v : encSize) {
        // Reported compressed sizes in bytes should be a multiple of 16 for
        // aligned packing
        EXPECT_EQ(v % 16, 0);
    }

    auto dec_dev = res.alloc<uint8_t>(stream, numInBatch * inBatchSize);
    auto outSuccess_dev = res.alloc<uint8_t>(stream, numInBatch);
    auto outSize_dev = res.alloc<uint32_t>(stream, numInBatch);

    // FIXME: Copy the compressed data to the host and truncate it to make
    // sure the compressed size is accurate
    ansDecodeBatchStride(res, ANSCodecConfig(prec, true), numInBatch,
                         enc_dev.data(), outBatchStride, dec_dev.data(),
                         inBatchSize, inBatchSize, outSuccess_dev.data(),
                         outSize_dev.data(), stream);

    auto outSuccess = outSuccess_dev.copyToHost(stream);
    auto outSize = outSize_dev.copyToHost(stream);

    for (auto s : outSuccess) {
        EXPECT_TRUE(s);
    }

    for (auto s : outSize) {
        EXPECT_EQ(s, inBatchSize);
    }

    auto dec = dec_dev.copyToHost(stream);
    EXPECT_EQ(orig, dec);
}

TEST(ANSTest, ZeroSized) {
    auto res = makeStackMemory();
    runBatchPointer(res, 10, {0}, 10.0);
}

TEST(ANSTest, BatchPointer) {
    auto res = makeStackMemory();

    // This build only instantiates kANSDefaultProbBits; see
    // ANSCodecConfig::probBits for why 9 and 11 are no longer exercised.
    const int prec = kANSDefaultProbBits;
    for (auto lambda : {1.0, 10.0, 100.0, 1000.0}) {
        runBatchPointer(res, prec, {1}, lambda);
        runBatchPointer(res, prec, {1, 1}, lambda);
        runBatchPointer(res, prec, {4096, 4095, 4096}, lambda);
        runBatchPointer(res, prec, {1234, 2345, 3456}, lambda);
        runBatchPointer(res, prec, {10000, 10013, 10000}, lambda);
    }
}

TEST(ANSTest, BatchPointerLarge) {
    auto res = makeStackMemory();

    std::random_device rd;
    std::mt19937 gen(10);
    std::uniform_int_distribution<uint32_t> dist(100, 10000);

    std::vector<uint32_t> sizes;
    for (int i = 0; i < 100; ++i) {
        sizes.push_back(dist(gen));
    }

    runBatchPointer(res, 10, sizes);
}

TEST(ANSTest, BatchStride) {
    auto res = makeStackMemory();

    // FIXME: 16 byte alignment required
    runBatchStride(res, 10, 13, 8192 + 16);
}

TEST(ANSTest, SingleKernel) {
    auto res = makeStackMemory();

    const int prec = kANSDefaultProbBits;

    // The cooperative path differs from the multi-launch one in ways only
    // certain shapes exercise: a member with no blocks still needs a header, a
    // sub-block member takes the partial-block branch while an exact multiple
    // of the 4 KiB block does not, and anything large makes the encode phase
    // loop over blocks grid-stride instead of one warp owning one block.
    auto batchSets = std::vector<std::vector<uint32_t>>{
        {0},
        {1},
        {4095},
        {4096},
        {4097},
        {1000, 4096, 100000, 1},
        {1 << 20, 1 << 20},
        {1 << 22},
        // More blocks than the co-resident grid has warps, so the encode phase
        // has to loop grid-stride instead of one warp owning one block. 24 MiB
        // is 6144 blocks against a grid of at most a few thousand warps.
        {24 << 20},
    };

    // More members than any device can keep co-resident, so the cooperative
    // launch would be rejected: this has to fall back to the multi-launch path
    // and still produce a correct archive, not fail.
    batchSets.push_back(std::vector<uint32_t>(1024, 512));

    for (auto& sizes : batchSets) {
        for (auto lambda : {1.0, 100.0}) {
            auto single = runBatchPointer(res, prec, sizes, lambda, true);
            auto multi = runBatchPointer(res, prec, sizes, lambda, false);

            // The two paths must agree on every member's compressed size. Their
            // byte layouts need not match -- each block reserves its home from
            // a shared tail, so blocks land in warp completion order -- but the
            // histogram is an exact integer sum and the normalization is the
            // same function at the same thread count, so the per-block lengths,
            // and therefore the totals, are identical.
            EXPECT_EQ(single, multi);
        }
    }
}
