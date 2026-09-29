#ifndef EXASIM_DISTRIBUTED_RADIX_QUANTILES_HPP
#define EXASIM_DISTRIBUTED_RADIX_QUANTILES_HPP

#include <Kokkos_BitManipulation.hpp>
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace exasim_meshadapt {
namespace detail {

template <typename Scalar>
using RadixKey = typename std::conditional<sizeof(Scalar) == sizeof(std::uint32_t),
                                           std::uint32_t, std::uint64_t>::type;

template <typename Scalar, typename Index>
inline void DistributedRadixQuantilesImpl(
    Scalar& hmin, Scalar& hmax, const Scalar* values, Index localCount,
    Scalar qmin, Scalar qmax
#ifdef HAVE_MPI
    , MPI_Comm communicator
#endif
)
{
    // Mesh means are positive and finite, so their unsigned IEEE bit patterns
    // have the same order as their values. Each pass retains one key byte and
    // communicates only two 256-bin histograms, never the element values.
    static_assert(std::is_same<Scalar, float>::value ||
                  std::is_same<Scalar, double>::value,
                  "DistributedRadixQuantiles supports float and double only.");
    if (localCount < 0)
        throw std::invalid_argument("DistributedRadixQuantiles: localCount is negative.");
    if (!(qmin >= Scalar(0) && qmin <= Scalar(1) &&
          qmax >= Scalar(0) && qmax <= Scalar(1) && qmin <= qmax))
        throw std::invalid_argument("DistributedRadixQuantiles: invalid quantile interval.");

    std::uint64_t globalCount = static_cast<std::uint64_t>(localCount);
#ifdef HAVE_MPI
    const std::uint64_t localCount64 = globalCount;
    MPI_Allreduce(&localCount64, &globalCount, 1, MPI_UINT64_T, MPI_SUM,
                  communicator);
#endif
    if (globalCount == 0)
        throw std::invalid_argument("DistributedRadixQuantiles: no input values.");

    auto quantileIndex = [globalCount](Scalar q) {
        std::uint64_t index = static_cast<std::uint64_t>(
            std::round(q*static_cast<Scalar>(globalCount)));
        if (index > 0) --index;
        return index < globalCount ? index : globalCount - 1;
    };

    using Key = RadixKey<Scalar>;
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = typename execution_space::memory_space;
    using unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    using input_view = Kokkos::View<const Scalar*, memory_space, unmanaged>;
    using count_view = Kokkos::View<std::uint64_t*, memory_space>;
    using key_view = Kokkos::View<Key*, memory_space>;

    constexpr int bins = 256;
    constexpr int selectors = 2;
    constexpr int histogramSize = selectors*bins;
    constexpr int keyBytes = sizeof(Key);
    constexpr int keyBits = 8*keyBytes;
    constexpr Index valuesPerTeam = 4096;

    input_view input(values, localCount);
    count_view histogram("distributed_radix_histogram", histogramSize);
    key_view prefixes("distributed_radix_prefixes", selectors);
    count_view ranks("distributed_radix_ranks", selectors);
    Kokkos::View<Scalar*, memory_space> result("distributed_radix_result", selectors);

    auto hostPrefixes = Kokkos::create_mirror_view(prefixes);
    auto hostRanks = Kokkos::create_mirror_view(ranks);
    hostPrefixes(0) = Key(0);
    hostPrefixes(1) = Key(0);
    hostRanks(0) = quantileIndex(qmin);
    hostRanks(1) = quantileIndex(qmax);
    Kokkos::deep_copy(prefixes, hostPrefixes);
    Kokkos::deep_copy(ranks, hostRanks);

    using team_policy = Kokkos::TeamPolicy<execution_space>;
    using member_type = typename team_policy::member_type;
    const Index leagueSize = (localCount + valuesPerTeam - 1)/valuesPerTeam;
    const std::size_t scratchBytes = histogramSize*sizeof(std::uint32_t);

    for (int pass = 0; pass < keyBytes; ++pass) {
        Kokkos::deep_copy(histogram, std::uint64_t(0));
        const int shift = keyBits - 8*(pass + 1);
        const Key prefixMask = pass == 0
            ? Key(0)
            : std::numeric_limits<Key>::max() << (keyBits - 8*pass);

        if (leagueSize > 0) {
            team_policy policy(leagueSize, Kokkos::AUTO());
            policy.set_scratch_size(0, Kokkos::PerTeam(scratchBytes));
            Kokkos::parallel_for(
                "distributed_radix_histogram", policy,
                KOKKOS_LAMBDA(const member_type& team) {
                    auto* localHistogram = static_cast<std::uint32_t*>(
                        team.team_shmem().get_shmem(scratchBytes));
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, histogramSize),
                        [&](const int i) { localHistogram[i] = 0; });
                    team.team_barrier();

                    const Index begin = static_cast<Index>(team.league_rank())*
                                        valuesPerTeam;
                    const Index end = begin + valuesPerTeam < localCount
                        ? begin + valuesPerTeam : localCount;
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, end - begin),
                        [&](const Index offset) {
                            const Key key = Kokkos::bit_cast<Key>(input(begin + offset));
                            const int bin = static_cast<int>((key >> shift) & Key(0xff));
                            for (int selector = 0; selector < selectors; ++selector)
                                if ((key & prefixMask) == prefixes(selector))
                                    Kokkos::atomic_inc(
                                        &localHistogram[selector*bins + bin]);
                        });
                    team.team_barrier();

                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, histogramSize),
                        [&](const int i) {
                            if (localHistogram[i] != 0)
                                Kokkos::atomic_add(&histogram(i),
                                    static_cast<std::uint64_t>(localHistogram[i]));
                        });
                });
            Kokkos::fence();
        }

#ifdef HAVE_MPI
        MPI_Allreduce(MPI_IN_PLACE, histogram.data(), histogramSize,
                      MPI_UINT64_T, MPI_SUM, communicator);
#endif

        Kokkos::parallel_for(
            "distributed_radix_select", Kokkos::RangePolicy<execution_space>(0, selectors),
            KOKKOS_LAMBDA(const int selector) {
                const std::uint64_t target = ranks(selector);
                std::uint64_t preceding = 0;
                for (int bin = 0; bin < bins; ++bin) {
                    const std::uint64_t count = histogram(selector*bins + bin);
                    if (target < preceding + count) {
                        prefixes(selector) |= static_cast<Key>(bin) << shift;
                        ranks(selector) = target - preceding;
                        break;
                    }
                    preceding += count;
                }
            });
    }

    Kokkos::parallel_for(
        "distributed_radix_result", Kokkos::RangePolicy<execution_space>(0, selectors),
        KOKKOS_LAMBDA(const int selector) {
            result(selector) = Kokkos::bit_cast<Scalar>(prefixes(selector));
        });
    const auto hostResult = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), result);
    hmin = hostResult(0);
    hmax = hostResult(1);
}

} // namespace detail

inline void DistributedRadixQuantiles(
    dstype& hmin, dstype& hmax, const dstype* values, Int localCount,
    dstype qmin, dstype qmax)
{
    detail::DistributedRadixQuantilesImpl(hmin, hmax, values, localCount,
                                          qmin, qmax
#ifdef HAVE_MPI
                                          , EXASIM_COMM_WORLD
#endif
    );
}

} // namespace exasim_meshadapt

#endif
