/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <glog/logging.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "dietgpu/ans/BatchProvider.cuh"
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/float/GpuFloatDecompress.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/StackDeviceMemory.h"

namespace dietgpu {

FloatDecompressStatus floatDecompress(StackDeviceMemory& res,
                                      const FloatDecompressConfig& config,
                                      uint32_t numInBatch, const void** in,
                                      void** out, const uint32_t* outCapacity,
                                      uint8_t* outSuccess_dev,
                                      uint32_t* outSize_dev,
                                      cudaStream_t stream) {
    // If the batch size is <= kBSLimit, we avoid cudaMemcpy and send all data
    // at kernel launch
    constexpr int kLimit = 128;

    // Decoding is a single pass for any output alignment; see
    // floatDecompressDevice. What the single-pass writer does require is that
    // each output pointer be aligned to the float word size, because it stores
    // one word at a time. That has always been the API's contract, but it was
    // assumed rather than checked, and a violation would surface as a
    // device-side misaligned access fault with nothing pointing back at this
    // argument.
    auto floatWordSize = getWordSizeFromFloatType(config.floatType);

    for (int i = 0; i < numInBatch; ++i) {
        CHECK(isPointerAligned(out[i], floatWordSize))
            << "float decompression output " << i << " (" << out[i]
            << ") must be aligned to the float word size " << floatWordSize;
    }

    if (numInBatch <= kLimit) {
        // We can do everything in a single pass without a h2d memcpy
        auto inProvider =
            BatchProviderInlinePointer<kLimit>(numInBatch, (void**)in);
        auto outProvider = BatchProviderInlinePointerCapacity<kLimit>(
            numInBatch, out, outCapacity);

        return floatDecompressDevice(res, config, numInBatch, inProvider,
                                     outProvider, outSuccess_dev, outSize_dev,
                                     stream);
    }

    // Copy data to device
    // To reduce latency, we prefer to coalesce all data together and copy as
    // one contiguous chunk
    static_assert(sizeof(void*) == sizeof(uintptr_t));
    static_assert(sizeof(uint32_t) <= sizeof(uintptr_t));

    // in, out, outCapacity
    auto params_dev = res.alloc<uintptr_t>(stream, numInBatch * 3);
    auto params_host =
        std::unique_ptr<uintptr_t[]>(new uintptr_t[3 * numInBatch]);

    std::memcpy(&params_host[0], in, numInBatch * sizeof(void*));
    std::memcpy(&params_host[numInBatch], out, numInBatch * sizeof(void*));
    std::memcpy(&params_host[2 * numInBatch], outCapacity,
                numInBatch * sizeof(uint32_t));

    CUDA_VERIFY(cudaMemcpyAsync(params_dev.data(), params_host.get(),
                                3 * numInBatch * sizeof(uintptr_t),
                                cudaMemcpyHostToDevice, stream));

    auto in_dev = params_dev.data();
    auto out_dev = params_dev.data() + numInBatch;
    auto outCapacity_dev =
        (const uint32_t*)(params_dev.data() + 2 * numInBatch);

    auto inProvider = BatchProviderPointer((void**)in_dev);
    auto outProvider = BatchProviderPointer((void**)out_dev, outCapacity_dev);

    return floatDecompressDevice(res, config, numInBatch, inProvider,
                                 outProvider, outSuccess_dev, outSize_dev,
                                 stream);
}

FloatDecompressStatus floatDecompressSplitSize(
    StackDeviceMemory& res, const FloatDecompressConfig& config,
    uint32_t numInBatch, const void** in, void* out,
    const uint32_t* outSplitSizes, uint8_t* outSuccess_dev,
    uint32_t* outSize_dev, cudaStream_t stream) {
    auto floatWordSize = getWordSizeFromFloatType(config.floatType);

    // Concatenate splitSize and splitSizePrefix together for a single h2d copy
    auto splitSizeHost = std::vector<uint32_t>(numInBatch * 2);
    auto splitSize = splitSizeHost.data();
    auto splitSizePrefix = splitSizeHost.data() + numInBatch;

    // As in floatDecompress: the single-pass writer stores one float word at a
    // time, so the base pointer has to be float word aligned. Interior splits
    // cannot break that, since each member starts at splitSizePrefix[i]
    // elements and so is a whole number of words past an aligned base -- which
    // is also why the 16 byte interior split check that used to live here is
    // gone.
    CHECK(isPointerAligned(out, floatWordSize))
        << "float decompression output (" << out
        << ") must be aligned to the float word size " << floatWordSize;

    for (uint32_t i = 0; i < numInBatch; ++i) {
        splitSize[i] = outSplitSizes[i];
        if (i > 0) {
            splitSizePrefix[i] = splitSizePrefix[i - 1] + splitSize[i - 1];
        }
    }

    auto sizes_dev = res.copyAlloc<uint32_t>(stream, splitSizeHost);

    // FIXME: combine with above for a single h2d copy
    auto in_dev = res.copyAlloc<void*>(stream, (void**)in, numInBatch);

    auto inProvider = BatchProviderPointer(in_dev.data());

    auto outProvider = BatchProviderSplitSize(
        out, sizes_dev.data(), sizes_dev.data() + numInBatch, floatWordSize);

    return floatDecompressDevice(res, config, numInBatch, inProvider,
                                 outProvider, outSuccess_dev, outSize_dev,
                                 stream);
}

}  // namespace dietgpu
