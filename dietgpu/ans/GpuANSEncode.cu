/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "dietgpu/ans/BatchProvider.cuh"
#include "dietgpu/ans/GpuANSEncode.cuh"

namespace dietgpu {

uint32_t getMaxCompressedSize(uint32_t uncompressedBytes) {
    uint32_t blocks = divUp(uncompressedBytes, kDefaultBlockSize);

    // The archive overhead grows with the block count -- one ANSWarpState and
    // one uint2 of block words per block -- so it has to be asked for `blocks`
    // blocks. It used to be asked for kDefaultBlockSize (4096) blocks, which
    // made the overhead a constant 557600 bytes regardless of the input: a
    // ~545 KiB over-estimate below 16 MiB, and above 4096 blocks an
    // under-estimate of 136 bytes per block (557 KiB at 32 MiB, 35 MiB at 1
    // GiB) in the buffer the caller sizes from this bound. The shortfall is not
    // reachable by real data -- it needs every block to expand to 1.217x, while
    // a 256 symbol / 10 bit table caps the average below 1.0x -- but a bound
    // that is wrong in the unsafe direction is not a bound.
    size_t rawSize = ANSCoalescedHeader::getCompressedOverhead(blocks);
    rawSize += (size_t)getMaxBlockSizeCoalesced(kDefaultBlockSize) * blocks;

    // When used in batches, we must align everything to 16 byte boundaries (due
    // to uint4 read/writes)
    rawSize = roundUp(rawSize, sizeof(uint4));
    CHECK_LE(rawSize, std::numeric_limits<int32_t>::max());

    return rawSize;
}

void ansEncodeBatchStride(StackDeviceMemory& res, const ANSCodecConfig& config,
                          uint32_t numInBatch, const void* in_dev,
                          uint32_t inPerBatchSize, uint32_t inPerBatchStride,
                          const uint32_t* histogram_dev, void* out_dev,
                          uint32_t outPerBatchStride, uint32_t* outSize_dev,
                          cudaStream_t stream) {
    auto inProvider =
        BatchProviderStride((void*)in_dev, inPerBatchStride, inPerBatchSize);
    auto outProvider = BatchProviderStride(out_dev, outPerBatchStride);

    ansEncodeBatchDevice(res, config, numInBatch, inProvider, histogram_dev,
                         inPerBatchSize,  // max size
                         outProvider, outSize_dev, stream);
}

void ansEncodeBatchPointer(StackDeviceMemory& res, const ANSCodecConfig& config,
                           uint32_t numInBatch, const void** in,
                           const uint32_t* inSize,
                           const uint32_t* histogram_dev, void** out,
                           uint32_t* outSize_dev, cudaStream_t stream) {
    // Get the total and maximum input size
    uint32_t maxSize = 0;

    for (uint32_t i = 0; i < numInBatch; ++i) {
        uint32_t curSize = inSize[i] / sizeof(ANSDecodedT);
        maxSize = std::max(maxSize, curSize);
    }

    // If the batch size is <= kBSLimit, we avoid cudaMemcpy and send all data
    // at kernel launch. These three copies are from *pageable* host memory, so
    // beyond their ~1 us of host time each they also force the runtime to stage
    // the transfer instead of just enqueueing it. ansDecodeBatchPointer has
    // done this for a long time; the encode side did not.
    constexpr int kBSLimit = 128;

    if (numInBatch <= kBSLimit) {
        auto inProvider = BatchProviderInlinePointerCapacity<kBSLimit>(
            numInBatch, (void**)in, inSize);
        auto outProvider =
            BatchProviderInlinePointer<kBSLimit>(numInBatch, out);

        ansEncodeBatchDevice(res, config, numInBatch, inProvider, histogram_dev,
                             maxSize, outProvider, outSize_dev, stream);

        return;
    }

    // Otherwise, we have to perform h2d copies
    auto in_dev = res.alloc<void*>(stream, numInBatch);
    auto inSize_dev = res.alloc<uint32_t>(stream, numInBatch);
    auto out_dev = res.alloc<void*>(stream, numInBatch);

    CUDA_VERIFY(cudaMemcpyAsync(in_dev.data(), in, numInBatch * sizeof(void*),
                                cudaMemcpyHostToDevice, stream));

    CUDA_VERIFY(cudaMemcpyAsync(inSize_dev.data(), inSize,
                                numInBatch * sizeof(uint32_t),
                                cudaMemcpyHostToDevice, stream));

    CUDA_VERIFY(cudaMemcpyAsync(out_dev.data(), out, numInBatch * sizeof(void*),
                                cudaMemcpyHostToDevice, stream));

    auto inProvider =
        BatchProviderPointer((void**)in_dev.data(), inSize_dev.data());
    auto outProvider = BatchProviderPointer(out_dev.data());

    ansEncodeBatchDevice(res, config, numInBatch, inProvider, histogram_dev,
                         maxSize, outProvider, outSize_dev, stream);
}

void ansEncodeBatchSplitSize(StackDeviceMemory& res,
                             const ANSCodecConfig& config, uint32_t numInBatch,
                             const void* in_dev, const uint32_t* inSplitSizes,
                             const uint32_t* histogram_dev, void* out_dev,
                             uint32_t outStride, uint32_t* outSize_dev,
                             cudaStream_t stream) {
    auto splitSizeHost = std::vector<uint32_t>(numInBatch * 2);
    auto splitSize = splitSizeHost.data();
    auto splitSizePrefix = splitSizeHost.data() + numInBatch;
    uint32_t maxSplitSize = 0;

    // check alignment
    CHECK_EQ(uintptr_t(in_dev) % kANSRequiredAlignment, 0);

    for (uint32_t i = 0; i < numInBatch; ++i) {
        auto size = inSplitSizes[i];

        if (i != (numInBatch - 1)) {
            // check alignment (internal splits affect alignment of things after
            // it)
            CHECK_EQ(size % kANSRequiredAlignment, 0);
        }

        splitSize[i] = size;
        if (i > 0) {
            splitSizePrefix[i] = splitSizePrefix[i - 1] + splitSize[i - 1];
        }

        maxSplitSize = std::max(size, maxSplitSize);
    }

    // Copy data to device
    // splitSize, splitSizePrefix
    auto sizes_dev = res.alloc<uint32_t>(stream, splitSizeHost.size());

    CUDA_VERIFY(cudaMemcpyAsync(sizes_dev.data(), splitSizeHost.data(),
                                splitSizeHost.size() * sizeof(uint32_t),
                                cudaMemcpyHostToDevice, stream));

    auto inProvider =
        BatchProviderSplitSize((void*)in_dev, sizes_dev.data(),
                               sizes_dev.data() + numInBatch, sizeof(uint8_t));

    auto outProvider = BatchProviderStride(out_dev, outStride);

    ansEncodeBatchDevice(res, config, numInBatch, inProvider, histogram_dev,
                         maxSplitSize, outProvider, outSize_dev, stream);
}

}  // namespace dietgpu
