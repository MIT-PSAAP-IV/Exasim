#include <Kokkos_Core.hpp>
#include "../../backend/Common/common.h"
#include "../../backend/Common/kokkosimpl.h"
#include "../../backend/Common/auxiliary_snapshot.hpp"
#include <limits>
#include <cstdio>

template <class T> int finiteChecks()
{
    int failures = 0;
    failures += !is_finite_bitwise(T(0));
    failures += !is_finite_bitwise(T(-1)); // auxiliary variables need not be positive
    failures += !is_finite_bitwise(std::numeric_limits<T>::max());
    failures += is_finite_bitwise(std::numeric_limits<T>::infinity());
    failures += is_finite_bitwise(-std::numeric_limits<T>::infinity());
    failures += is_finite_bitwise(std::numeric_limits<T>::quiet_NaN());
    return failures;
}

template <class T> int snapshots()
{
    int failures = 0;
    Kokkos::View<T*> w("w", 4);
    ArraySetValue(w.data(), T(2), 4);
    {
        AuxiliaryStateSnapshot<T> base(w.data(), 4);
        ArraySetValue(w.data(), std::numeric_limits<T>::quiet_NaN(), 4);
        base.restore();
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w);
        for (int i=0; i<4; ++i) failures += !is_finite_bitwise(h(i)) || h(i)!=T(2);
        // Nested FD probes must not replace the enclosing trial's initial seed.
        {
            AuxiliaryStateSnapshot<T> probe(w.data(), 4);
            ArraySetValue(w.data(), T(7), 4);
        }
        h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w);
        for (int i=0; i<4; ++i) failures += h(i)!=T(2);
        ArraySetValue(w.data(), T(-3), 4);
        base.commit();
    }
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w);
    for (int i=0; i<4; ++i) failures += h(i)!=T(-3);
    { AuxiliaryStateSnapshot<T> empty(nullptr, 0); empty.restore(); }
    return failures;
}

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    int failures = finiteChecks<float>() + finiteChecks<double>();
    failures += snapshots<float>() + snapshots<double>();
    Kokkos::finalize();
    std::printf("auxiliary trial safeguards: %d failures\n", failures);
    return failures ? 1 : 0;
}
