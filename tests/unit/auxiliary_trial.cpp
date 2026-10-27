#include <Kokkos_Core.hpp>
#include "../../backend/Common/common.h"
#include "../../backend/Common/kokkosimpl.h"
#include <limits>
#include <cstdio>
#include <cmath>

static int snapshotAllocations = 0;

// Runtime bit patterns and arithmetic-derived norms exercise the optimized
// call sites, not just checks that can be evaluated at compile time.
template <class T> int runtimeFiniteChecks()
{
    using Bits = std::conditional_t<std::is_same_v<T, double>, std::uint64_t, std::uint32_t>;
    constexpr Bits exponent = std::is_same_v<T, double> ? 0x7ff0000000000000ULL : 0x7f800000U;
    constexpr Bits sign = std::is_same_v<T, double> ? 0x8000000000000000ULL : 0x80000000U;
    volatile Bits patterns[] = {0, sign, exponent - 1, exponent, sign | exponent, exponent | 1};
    int failures = 0;
    for (int i = 0; i < 6; ++i) {
        Bits bits = patterns[i];
        T value;
        std::memcpy(&value, &bits, sizeof(value));
        failures += is_finite_bitwise(value) != (i < 3);
        T norm = std::sqrt(value);
        failures += is_finite_bitwise(norm) != (i < 3);
        // Match both outer rejection and finite-step acceptance conditions.
        const bool reject = !is_finite_bitwise(norm) || norm > T(1e6);
        const bool accept = is_finite_bitwise(norm) && norm <= T(1e6);
        if (i >= 3) failures += !reject || accept;
    }
    volatile T negative = T(-1);
    T norm = std::sqrt(negative);
    failures += is_finite_bitwise(norm);
    return failures;
}

template <class T> int finiteChecks()
{
    int failures = 0;
    failures += !is_finite_bitwise(T(0));
    failures += !is_finite_bitwise(T(-1)); // auxiliary variables need not be positive
    failures += !is_finite_bitwise(std::numeric_limits<T>::max());
    failures += is_finite_bitwise(bitwise_infinity<T>());
    failures += is_finite_bitwise(bitwise_infinity<T>(true));
    failures += is_finite_bitwise(bitwise_quiet_nan<T>());
    return failures;
}

template <class T> int snapshots()
{
    int failures = 0;
    Kokkos::View<T*> w("w", 4);
    ArraySetValue(w.data(), T(2), 4);
    {
        AuxiliaryStateSnapshot<T> base(w.data(), 4);
        ArraySetNaN(w.data(), 4);
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

template <class T> int nanMarkers()
{
    using Bits=std::conditional_t<std::is_same_v<T,double>,std::uint64_t,std::uint32_t>;
    const Bits expected=std::is_same_v<T,double> ? Bits(0x7ff8000000000000ULL) : Bits(0x7fc00000U);
    Kokkos::View<T*> w("nan_marker_test",19);
    ArraySetValue(w.data(),T(-1),19);
    ArraySetNaN(w.data()+1,17); // exercise tail length and preserve both guards
    auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),w);
    int failures=(host(0)!=T(-1))+(host(18)!=T(-1));
    for (int i=1;i<18;++i) {
        Bits bits;
        std::memcpy(&bits,&host(i),sizeof(T));
        failures+=bits!=expected;
    }
    ArraySetNaN(w.data(),0);
    ArraySetValue(w.data(),T(2),19);
    host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),w);
    for (int i=0;i<19;++i) failures+=host(i)!=T(2);
    return failures;
}

template <class T> int cachedSnapshots()
{
    int failures = 0;
    tempstructT<T> workspace;
    auto& cache = workspace.auxiliaryMatvecBackup;
    Kokkos::View<T*> w("cached_w", 16);
    ArraySetValue(w.data(), T(-9), 16);
    const int initialAllocations = snapshotAllocations;
    { AuxiliaryStateSnapshot<T> empty(nullptr, 0, &cache); }
    failures += cache.data() != nullptr || snapshotAllocations != initialAllocations;

    // Fresh values are saved each time, but the allocation is retained.
    T* firstBuffer = nullptr;
    for (int trial = 0; trial < 20; ++trial) {
        const T seed = T(trial + 1);
        ArraySetValue(w.data(), seed, 4);
        {
            AuxiliaryStateSnapshot<T> probe(w.data(), 4, &cache);
            if (trial == 0) firstBuffer = cache.data();
            failures += cache.data() != firstBuffer;
            ArraySetNaN(w.data(), 4);
        }
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w);
        for (int i=0; i<16; ++i)
            failures += !is_finite_bitwise(h(i)) || h(i) != (i<4 ? seed : T(-9));
    }
    failures += snapshotAllocations != initialAllocations + 1;
    failures += workspace.sizeoffloat() != 4;

    // Re-entering the same cache must not overwrite or resize a live backup.
    ArraySetValue(w.data(), T(2), 16);
    {
        AuxiliaryStateSnapshot<T> outer(w.data(), 4, &cache);
        ArraySetValue(w.data(), T(7), 4);
        {
            AuxiliaryStateSnapshot<T> nested(w.data(), 4, &cache);
            ArraySetNaN(w.data(), 4);
        }
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w);
        for (int i=0; i<4; ++i) failures += !is_finite_bitwise(h(i)) || h(i) != T(7);
        {
            AuxiliaryStateSnapshot<T> larger(w.data(), 16, &cache);
            ArraySetNaN(w.data(), 16);
        }
        h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w);
        for (int i=0; i<16; ++i)
            failures += !is_finite_bitwise(h(i)) || h(i) != (i<4 ? T(7) : T(2));
        failures += cache.data() != firstBuffer || cache.extent(0) != 4;
    }
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w);
    for (int i=0; i<16; ++i) failures += !is_finite_bitwise(h(i)) || h(i) != T(2);
    failures += snapshotAllocations != initialAllocations + 3;

    // Enclosing Newton/RB snapshots keep independent storage, as in the solver.
    {
        AuxiliaryStateSnapshot<T> outer(w.data(), 16);
        ArraySetValue(w.data(), T(5), 4);
        {
            AuxiliaryStateSnapshot<T> probe(w.data(), 4, &cache);
            ArraySetValue(w.data(), T(8), 4);
            probe.restore();
            probe.commit();
        }
    }
    h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w);
    for (int i=0; i<16; ++i) failures += !is_finite_bitwise(h(i)) || h(i) != T(2);

    // Grow only when necessary; shrinking the active extent must not reallocate.
    const int beforeGrowth = snapshotAllocations;
    {
        AuxiliaryStateSnapshot<T> larger(w.data(), 16, &cache);
        ArraySetValue(w.data(), T(-3), 16);
        larger.commit();
    }
    failures += cache.extent(0) != 16 || snapshotAllocations != beforeGrowth + 1;
    firstBuffer = cache.data();
    {
        AuxiliaryStateSnapshot<T> smaller(w.data(), 4, &cache);
        ArraySetNaN(w.data(), 4);
    }
    h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w);
    for (int i=0; i<16; ++i) failures += !is_finite_bitwise(h(i)) || h(i) != T(-3);
    failures += cache.data() != firstBuffer || snapshotAllocations != beforeGrowth + 1;
    failures += workspace.sizeoffloat() != 16;
    workspace.freememory(0);
    failures += cache.data() != nullptr || workspace.sizeoffloat() != 0;
    return failures;
}

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    if (argc>1 && std::strcmp(argv[1],"bad-extent")==0) {
        Kokkos::View<double*> w("bad_extent",1);
        AuxiliaryStateSnapshot<double> snapshot(w.data(),0);
        return 0; // test must fail before reaching here
    }
    if (argc>1 && std::strcmp(argv[1],"null-state")==0) {
        AuxiliaryStateSnapshot<double> snapshot(nullptr,1);
        return 0;
    }
    Kokkos::Tools::Experimental::set_allocate_data_callback(
        [](Kokkos::Tools::SpaceHandle, const char* label, const void*, const uint64_t) {
            if (std::strcmp(label, "auxiliary_snapshot") == 0) ++snapshotAllocations;
        });
    int failures = finiteChecks<float>() + finiteChecks<double>();
    failures += runtimeFiniteChecks<float>() + runtimeFiniteChecks<double>();
    failures += snapshots<float>() + snapshots<double>();
    failures += nanMarkers<float>() + nanMarkers<double>();
    failures += cachedSnapshots<float>() + cachedSnapshots<double>();
    Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
    Kokkos::finalize();
    std::printf("auxiliary trial safeguards: %d failures\n", failures);
    return failures ? 1 : 0;
}
