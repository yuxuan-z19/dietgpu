/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <glog/logging.h>

#include <cmath>
#include <sstream>
#include <vector>

#include "dietgpu/ans/GpuANSCodec.h"
#include "dietgpu/ans/GpuANSDecode.cuh"
#include "dietgpu/ans/GpuANSUtils.cuh"
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/float/GpuFloatInfo.cuh"
#include "dietgpu/float/GpuFloatUtils.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/StackDeviceMemory.h"
#include "dietgpu/utils/StaticUtils.h"

namespace dietgpu {

template <FloatType FT, typename InProvider>
struct FloatANSProvider {
    using FTI = FloatTypeInfo<FT>;

    __host__ FloatANSProvider(InProvider& provider) : inProvider_(provider) {}

    __device__ void* getBatchStart(uint32_t batch) {
        uint8_t* p = (uint8_t*)inProvider_.getBatchStart(batch);

        // This is the first place that touches the header
        GpuFloatHeader h = *((GpuFloatHeader*)p);
        h.checkMagicAndVersion();
        assert(FT == h.getFloatType());

        // Increment the pointer to past the floating point data
        return p + sizeof(GpuFloatHeader) + FTI::getUncompDataSize(h.size);
    }

    __device__ const void* getBatchStart(uint32_t batch) const {
        const uint8_t* p = (const uint8_t*)inProvider_.getBatchStart(batch);

        // This is the first place that touches the header
        GpuFloatHeader h = *((const GpuFloatHeader*)p);
        h.checkMagicAndVersion();
        assert(FT == h.getFloatType());

        // Increment the pointer to past the floating point data
        return p + sizeof(GpuFloatHeader) + FTI::getUncompDataSize(h.size);
    }

    InProvider inProvider_;
};

template <FloatType FT, uint32_t BlockSize>
struct JoinFloatWriter {
    using FTI = FloatTypeInfo<FT>;

    __host__ __device__ JoinFloatWriter(uint32_t size, typename FTI::WordT* out,
                                        const typename FTI::NonCompT* nonComp)
        : out_(out),
          nonComp_(nonComp),
          outBlock_(nullptr),
          nonCompBlock_(nullptr) {}

    __host__ __device__ void setBlock(uint32_t block) {
        outBlock_ = out_ + block * BlockSize;
        nonCompBlock_ = nonComp_ + block * BlockSize;
    }

    __device__ void write(uint32_t offset, uint8_t sym) {
        auto nonComp = nonCompBlock_[offset];
        outBlock_[offset] = FTI::join(sym, nonComp);
    }

    typename FTI::WordT* out_;
    const typename FTI::NonCompT* nonComp_;
    typename FTI::WordT* outBlock_;
    const typename FTI::NonCompT* nonCompBlock_;
};

template <uint32_t BlockSize>
struct JoinFloatWriter<FloatType::kFloat32, BlockSize> {
    using FTI = FloatTypeInfo<FloatType::kFloat32>;

    __host__ __device__ JoinFloatWriter(uint32_t size, typename FTI::WordT* out,
                                        const typename FTI::NonCompT* nonComp)
        : size_(size),
          out_(out),
          nonComp_(nonComp),
          outBlock_(nullptr),
          nonCompBlock2_(nullptr),
          nonCompBlock1_(nullptr) {}

    __host__ __device__ void setBlock(uint32_t block) {
        nonCompBlock2_ = (const uint16_t*)nonComp_ + block * BlockSize;
        nonCompBlock1_ =
            (const uint8_t*)((const uint16_t*)nonComp_ + roundUp(size_, 8U)) +
            block * BlockSize;
        outBlock_ = out_ + block * BlockSize;
    }

    __device__ void write(uint32_t offset, uint8_t sym) {
        uint32_t nc = uint32_t(nonCompBlock1_[offset]) * 65536U +
                      uint32_t(nonCompBlock2_[offset]);

        outBlock_[offset] = FTI::join(sym, nc);
    }

    uint32_t size_;
    typename FTI::WordT* out_;
    const typename FTI::NonCompT* nonComp_;
    typename FTI::WordT* outBlock_;
    const uint16_t* nonCompBlock2_;
    const uint8_t* nonCompBlock1_;
};

template <typename InProvider, typename OutProvider, FloatType FT,
          uint32_t BlockSize>
struct FloatOutProvider {
    using Writer = JoinFloatWriter<FT, BlockSize>;
    using FTI = FloatTypeInfo<FT>;

    __host__ FloatOutProvider(InProvider& inProvider, OutProvider& outProvider)
        : inProvider_(inProvider), outProvider_(outProvider) {}

    __device__ void* getBatchStart(uint32_t batch) {
        return inProvider_.getBatchStart(batch);
    }

    __device__ uint32_t getBatchSize(uint32_t batch) {
        return outProvider_.getBatchSize(batch);
    }

    __device__ Writer getWriter(uint32_t batch) {
        // Get float header
        auto h = (const GpuFloatHeader*)getBatchStart(batch);

        return Writer(h->size,
                      (typename FTI::WordT*)outProvider_.getBatchStart(batch),
                      // advance past the header
                      (const typename FTI::NonCompT*)(h + 1));
    }

    InProvider inProvider_;
    OutProvider outProvider_;
};

template <typename InProvider, typename OutProvider>
FloatDecompressStatus floatDecompressDevice(
    StackDeviceMemory& res, const FloatDecompressConfig& config,
    uint32_t numInBatch, InProvider& inProvider, OutProvider& outProvider,
    uint8_t* outSuccess_dev, uint32_t* outSize_dev, cudaStream_t stream) {
    // not allowed in float mode
    assert(!config.ansConfig.useChecksum);

    // Decoding is always a single pass. The writer that rejoins each decoded
    // symbol with its stored remainder (JoinFloatWriter) is scalar, so it needs
    // only the float word alignment the API already requires of output
    // pointers, not 16 byte alignment.
    //
    // There used to be a second path for outputs that were not 16 byte aligned:
    // decode the exponents into a temporary the size of the whole uncompressed
    // output, then run a joinFloat kernel over it. That gate was vestigial --
    // the vectorized join it existed to protect is gone, and joinFloat picked
    // its scalar branch for precisely the unaligned outputs that were routed to
    // it. So the second path performed the same scalar stores, plus a full
    // extra round trip through device memory, plus a temporary as large as the
    // decompressed data.
#define RUN_FUSED(FT)                                                         \
    do {                                                                      \
        auto inProviderANS = FloatANSProvider<FT, InProvider>(inProvider);    \
        auto outProviderANS =                                                 \
            FloatOutProvider<InProvider, OutProvider, FT, kDefaultBlockSize>( \
                inProvider, outProvider);                                     \
                                                                              \
        ansDecodeBatch(res, config.ansConfig, numInBatch, inProviderANS,      \
                       outProviderANS, outSuccess_dev, outSize_dev, stream);  \
    } while (false)

    switch (config.floatType) {
        case FloatType::kFloat16:
            RUN_FUSED(FloatType::kFloat16);
            break;
        case FloatType::kBFloat16:
            RUN_FUSED(FloatType::kBFloat16);
            break;
        case FloatType::kFloat32:
            RUN_FUSED(FloatType::kFloat32);
            break;
        default:
            CHECK(false);
            break;
    }

#undef RUN_FUSED

    FloatDecompressStatus status;

    // Perform optional checksum, if desired
    if (config.useChecksum) {
        auto checksum_dev = res.alloc<uint32_t>(stream, numInBatch);
        auto sizes_dev = res.alloc<uint32_t>(stream, numInBatch);
        auto archiveChecksum_dev = res.alloc<uint32_t>(stream, numInBatch);

        // Checksum the output data
        checksumBatch(numInBatch, outProvider, checksum_dev.data(), stream);

        // Get prior checksum from the float headers
        floatGetCompressedInfo(inProvider, numInBatch, sizes_dev.data(),
                               nullptr, archiveChecksum_dev.data(), stream);

        // Compare against previously seen checksums on the host
        auto sizes = sizes_dev.copyToHost(stream);
        auto newChecksums = checksum_dev.copyToHost(stream);
        auto oldChecksums = archiveChecksum_dev.copyToHost(stream);

        std::stringstream errStr;

        for (int i = 0; i < numInBatch; ++i) {
            if (oldChecksums[i] != newChecksums[i]) {
                status.error = FloatDecompressError::ChecksumMismatch;

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
