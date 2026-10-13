#pragma once

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

namespace exasim::meshtransfer {

using DefaultScalar = double;
using DefaultIndex = std::int32_t;

enum PointStatus : int {
    FOUND = 0,
    LOCAL_NEXT = 1,
    REMOTE = 2,
    OUTSIDE = 3
};

enum class InverseMapStatus : int {
    Converged = 0,
    Nonconverged = 1,
    Singular = 2
};

class MeshTransferError : public std::runtime_error {
public:
    explicit MeshTransferError(const std::string& message)
        : std::runtime_error(message) {}
};

#ifdef HAVE_MPI
using Comm = MPI_Comm;
inline const Comm null_comm = MPI_COMM_NULL;
#else
using Comm = int;
inline constexpr Comm null_comm = 0;
#endif

template <class T>
struct MpiType;

#ifdef HAVE_MPI
template <> struct MpiType<float> { static MPI_Datatype value() { return MPI_FLOAT; } };
template <> struct MpiType<double> { static MPI_Datatype value() { return MPI_DOUBLE; } };
template <> struct MpiType<std::int32_t> { static MPI_Datatype value() {
    MPI_Datatype type = MPI_DATATYPE_NULL;
    MPI_Type_match_size(MPI_TYPECLASS_INTEGER, 4, &type);
    return type;
} };
template <> struct MpiType<std::int64_t> { static MPI_Datatype value() {
    MPI_Datatype type = MPI_DATATYPE_NULL;
    MPI_Type_match_size(MPI_TYPECLASS_INTEGER, 8, &type);
    return type;
} };
#endif

template <class Index>
Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<Index>>
range_policy(Index begin, Index end)
{
    return {begin, end};
}

} // namespace exasim::meshtransfer
