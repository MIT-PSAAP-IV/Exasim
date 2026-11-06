#pragma once

#include <Kokkos_Core.hpp>

#ifdef HAVE_ENZYME
// ClangEnzyme replaces these calls with derivative code during compilation.
// The executable does not need an Enzyme runtime library.
// This function template permits calls from CUDA and HIP device functions.
template <class... Args>
KOKKOS_FUNCTION void __enzyme_fwddiff(void*, Args...);

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
extern __device__ int enzyme_dup;
extern __device__ int enzyme_const;
#else
extern int enzyme_dup;
extern int enzyme_const;
#endif
#endif
