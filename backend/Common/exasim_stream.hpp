// SPDX-License-Identifier: see Exasim LICENSE
//
// exasim_stream.hpp -- the single HIP compute stream shared by Kokkos, hipBLAS and hand-launched kernels.
//
// Kokkos (>= 4) runs HIP kernels on a stream it creates itself. hipBLAS (handle never given a stream) and
// kernels launched with stream 0 use HIP's legacy null stream, which implicitly synchronizes with every
// blocking stream. Mixing the two therefore inserts a device-wide serialization point at every hipBLAS /
// MFMA call (measured on MI300A: ~12 us of idle GPU after each of ~14k MFMA launches per 3dcoarse solve).
// Putting all work on Kokkos' stream keeps the same kernels in the same order -- results are unchanged --
// without the cross-stream barriers.
//
// EXASIM_SINGLE_STREAM=0 restores the old null-stream behaviour (A/B testing).
#ifndef EXASIM_STREAM_HPP
#define EXASIM_STREAM_HPP

#ifdef HAVE_HIP
#include <Kokkos_Core.hpp>
#include <hip/hip_runtime.h>
#include <cstdlib>

namespace exasim_stream {

inline hipStream_t compute() {
    static const bool on = [](){ const char* e = std::getenv("EXASIM_SINGLE_STREAM"); return !(e && e[0] == '0'); }();
    if (!on || !Kokkos::is_initialized()) return 0;
    return Kokkos::HIP().hip_stream();
}

} // namespace exasim_stream
#endif // HAVE_HIP
#endif // EXASIM_STREAM_HPP
