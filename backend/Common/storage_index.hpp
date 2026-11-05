#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

// Mesh IDs and BLAS dimensions may remain int32. Storage extents/offsets must
// not inherit that type: a matrix can have >INT_MAX entries with few elements.
namespace exasim::index {
using Storage = std::size_t;
static_assert(sizeof(Storage) >= sizeof(std::uint64_t), "64-bit storage offsets required");

template<class T> inline Storage storageCount(std::uint64_t n, const char* name) {
    if (n > std::uint64_t(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(T))
        throw std::overflow_error(std::string("Storage byte extent exceeds pointer range: ") + name);
    return Storage(n);
}
template<class T> inline Storage storageProduct(std::uint64_t a, std::uint64_t b, const char* name) {
    const auto limit = std::uint64_t(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(T);
    if (a && b > limit / a)
        throw std::overflow_error(std::string("Storage product exceeds pointer range: ") + name);
    return Storage(a * b);
}
inline constexpr bool uses32BitLoop(Storage count) {
    // The loop's exclusive bound as well as every visited index must fit.
    return count <= Storage(std::numeric_limits<std::int32_t>::max());
}
template<class I> inline I loopCount(Storage count) {
    static_assert(std::is_same_v<I,std::int32_t> || std::is_same_v<I,std::int64_t>);
    if (count > Storage(std::numeric_limits<I>::max()))
        throw std::overflow_error("Loop extent exceeds selected index type");
    return I(count);
}
} // namespace exasim::index
