#ifndef EXASIM_FINITE_CHECKS_HPP
#define EXASIM_FINITE_CHECKS_HPP

#include <cstdint>
#include <cstring>
#include <type_traits>

// std::isfinite may be folded to true by the backend's -ffast-math build.
// These host-side checks are used on norms after their device/MPI reductions.
template <class T> inline bool is_finite_bitwise(T value)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    if constexpr (std::is_same_v<T, double>) {
        std::uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
    } else {
        std::uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return (bits & 0x7f800000U) != 0x7f800000U;
    }
}

#endif
