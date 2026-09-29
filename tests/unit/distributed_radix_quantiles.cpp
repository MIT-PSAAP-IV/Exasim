#ifdef HAVE_MPI
#include <mpi.h>
MPI_Comm EXASIM_COMM_WORLD = MPI_COMM_NULL;
MPI_Comm EXASIM_COMM_LOCAL = MPI_COMM_NULL;
#endif
#include <Kokkos_Core.hpp>
#include "../../backend/Common/common.h"
#include "../../backend/Solution/distributedradixquantiles.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

template <typename Scalar>
std::uint64_t quantileIndex(Scalar q, std::uint64_t count)
{
    std::uint64_t index = static_cast<std::uint64_t>(std::round(
        q*static_cast<Scalar>(count)));
    if (index > 0) --index;
    return index < count ? index : count - 1;
}

template <typename Scalar>
int checkCase(const std::string& name, const std::vector<Scalar>& globalValues,
              int rank, int ranks)
{
    const std::size_t begin = globalValues.size()*static_cast<std::size_t>(rank)/ranks;
    const std::size_t end = globalValues.size()*static_cast<std::size_t>(rank + 1)/ranks;
    Kokkos::View<Scalar*> local("radix_test_values", end - begin);
    auto hostLocal = Kokkos::create_mirror_view(local);
    for (std::size_t i = begin; i < end; ++i) hostLocal(i - begin) = globalValues[i];
    Kokkos::deep_copy(local, hostLocal);

    std::vector<Scalar> sorted = globalValues;
    std::sort(sorted.begin(), sorted.end());
    const Scalar quantiles[][2] = {
        {Scalar(0), Scalar(1)},
        {Scalar(0.2), Scalar(0.8)},
        {Scalar(0.5), Scalar(0.5)},
        {Scalar(0.03125), Scalar(0.96875)}
    };

    int failures = 0;
    for (const auto& q : quantiles) {
        Scalar actualMin = 0;
        Scalar actualMax = 0;
        exasim_meshadapt::detail::DistributedRadixQuantilesImpl(
            actualMin, actualMax, local.data(),
            static_cast<std::int64_t>(end - begin), q[0], q[1]
#ifdef HAVE_MPI
            , EXASIM_COMM_WORLD
#endif
        );
        const Scalar expectedMin = sorted[quantileIndex(q[0], sorted.size())];
        const Scalar expectedMax = sorted[quantileIndex(q[1], sorted.size())];
        if (actualMin != expectedMin || actualMax != expectedMax) {
            if (rank == 0)
                std::printf("FAIL %-18s q=(%.8g, %.8g): got (%.17g, %.17g), "
                            "std::sort gives (%.17g, %.17g)\n",
                            name.c_str(), static_cast<double>(q[0]),
                            static_cast<double>(q[1]), static_cast<double>(actualMin),
                            static_cast<double>(actualMax), static_cast<double>(expectedMin),
                            static_cast<double>(expectedMax));
            ++failures;
        }
    }
    return failures;
}

template <typename Scalar>
int checkPrecision(int rank, int ranks)
{
    int failures = 0;
    failures += checkCase<Scalar>("uniform", std::vector<Scalar>(257, Scalar(3.25)),
                                  rank, ranks);
    failures += checkCase<Scalar>("duplicates",
        {Scalar(9), Scalar(1), Scalar(4), Scalar(4), Scalar(2), Scalar(9),
         Scalar(0.5), Scalar(4), Scalar(7), Scalar(2)}, rank, ranks);
    failures += checkCase<Scalar>("empty partitions", {Scalar(8), Scalar(2)}, rank, ranks);

    std::mt19937_64 generator(8675309);
    std::uniform_real_distribution<double> mantissa(0.5, 1.0);
    std::uniform_int_distribution<int> exponent(-20, 20);
    std::vector<Scalar> randomValues(10007);
    for (Scalar& value : randomValues)
        value = static_cast<Scalar>(std::ldexp(mantissa(generator), exponent(generator)));
    failures += checkCase<Scalar>("random log-scale", randomValues, rank, ranks);
    return failures;
}

} // namespace

int main(int argc, char** argv)
{
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
    EXASIM_COMM_WORLD = MPI_COMM_WORLD;
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(EXASIM_COMM_WORLD, &rank);
    MPI_Comm_size(EXASIM_COMM_WORLD, &ranks);
#else
    const int rank = 0;
    const int ranks = 1;
#endif
    Kokkos::initialize(argc, argv);
    int failures = 0;
    {
        failures += checkPrecision<float>(rank, ranks);
        failures += checkPrecision<double>(rank, ranks);
        if (rank == 0 && failures == 0)
            std::printf("Distributed radix quantiles match std::sort for float and double.\n");
    }
    Kokkos::finalize();
#ifdef HAVE_MPI
    int globalFailures = 0;
    MPI_Allreduce(&failures, &globalFailures, 1, MPI_INT, MPI_SUM, EXASIM_COMM_WORLD);
    MPI_Finalize();
    return globalFailures == 0 ? 0 : 1;
#else
    return failures == 0 ? 0 : 1;
#endif
}
