#pragma once
#include "storage_index.hpp"
#include <hip/hip_runtime.h>

namespace exasim::index {
template<class I, class F> __global__ void forEachKernel(I count, F f) {
    // Promote before multiplying. Small loops retain one thread per entry;
    // large loops use a bounded grid and an overflow-safe grid stride.
    I i = I(blockIdx.x) * I(blockDim.x) + I(threadIdx.x);
    if constexpr (std::is_same_v<I,std::int32_t>) {
        if (i < count) f(i);
    } else {
        const I stride = I(gridDim.x) * I(blockDim.x);
        while (i < count) {
            f(i);
            if (count - i <= stride) break;
            i += stride;
        }
    }
}
template<class I, class F> inline hipError_t forEachHipAs(Storage count, hipStream_t stream, F f) {
    const I n = loopCount<I>(count);
    if (!n) return hipSuccess;
    const Storage blocks = 1 + (count - 1) / 256;
    const unsigned grid = unsigned(std::is_same_v<I,std::int32_t> ? blocks :
                                   (blocks < 65535 ? blocks : 65535));
    forEachKernel<I><<<grid,256,0,stream>>>(n,f);
    return hipGetLastError();
}
// F should accept auto (or an explicitly wide type) to avoid narrowing the
// dispatched index inside the callback. No synchronization or allocation.
template<class F> inline hipError_t forEachHip(Storage count, hipStream_t stream, F f) {
    if (uses32BitLoop(count)) return forEachHipAs<std::int32_t>(count,stream,f);
    return forEachHipAs<std::int64_t>(count,stream,f);
}
} // namespace exasim::index
