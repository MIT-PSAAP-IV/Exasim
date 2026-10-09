// Pre-included into the harness TU for the ROCm 7.0.0 (gpumpi) build only: clang-20 from ROCm 7.0.0 segfaults in the
// "correlated-propagation" pass on rocprim's radix_sort_single_helper, which Kokkos::sort instantiates when
// KOKKOS_ENABLE_ROCTHRUST is on. Exasim uses Kokkos::sort only outside the residual (setup/postprocessing), so the harness
// TU falls back to Kokkos' own sort. The production library itself was compiled with ROCm 6.3.1 and never hits this.
#pragma once
#include <Kokkos_Macros.hpp>
#undef KOKKOS_ENABLE_ROCTHRUST
