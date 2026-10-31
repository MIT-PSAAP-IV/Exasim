#pragma once
#include <Kokkos_Core.hpp>
#include <Kokkos_BitManipulation.hpp>
#include <cstdint>

namespace exasim::composite {
// Floating comparisons alone cannot detect NaNs under -ffast-math.
KOKKOS_INLINE_FUNCTION bool finiteValue(double value) {
    // The volatile integer round-trip prevents fast-math compilers from
    // replacing this bit test with a floating classification assumed true.
    const volatile auto bits = Kokkos::bit_cast<std::uint64_t>(value);
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}
KOKKOS_INLINE_FUNCTION bool invalidInverseValue(double value, bool packedFloat) {
    // Positive IEEE-754 binary64 magnitudes have the same integer ordering.
    // Clearing the sign also puts every NaN/Inf beyond both finite limits.
    // Keep the complete predicate in integer arithmetic under fast math.
    const auto magnitude = Kokkos::bit_cast<std::uint64_t>(value) & UINT64_C(0x7fffffffffffffff);
    const auto limit = packedFloat ? UINT64_C(0x47efffffe0000000) : UINT64_C(0x7e37e43c8800759c);
    return magnitude > limit;
}
} // namespace exasim::composite
